import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Translates index lookups into attribute lookups.
class AttrProxy(object):
	def __init__(self, module, prefix):
		self.module = module
		self.prefix = prefix

	def __getitem__(self, i):
		return getattr(self.module, self.prefix + str(i))

# depthwise separable convolution
class sep_conv(nn.Module):
	def __init__(self, nin, nout, kernel_size):
		super(sep_conv, self).__init__()
		self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,
									padding=1, groups=nin)
		self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

	def forward(self, x):
		out = self.depthwise(x)
		out = self.pointwise(out)
		return out

# downsample block
class down_block(nn.Module):
	def __init__(self, nin, nout, depthwise=True):
		super(down_block, self).__init__()
		self.conv1 = sep_conv(nin, nout, 3) if depthwise \
					else nn.Conv2d(nin, nout, 3, padding=1)
		self.lrelu1 = nn.LeakyReLU(0.2, True)
		self.conv2 = sep_conv(nout, nout, 3) if depthwise \
					else nn.Conv2d(nout, nout, 3, padding=1)
		self.lrelu2 = nn.LeakyReLU(0.2, True)
		self.bn = nn.BatchNorm2d(nout)
		self.pool = nn.AvgPool2d(2, stride=2)

	def forward(self, x):
		h = self.lrelu1(self.conv1(x))
		h = self.lrelu2(self.conv2(h))
		return self.bn(self.pool(h))

# upsample block
class up_block(nn.Module):
	def __init__(self, nin, nout, skip=False, depthwise=True):
		super(up_block, self).__init__()
		self.skip = skip
		c = 2 if skip else 1
		self.upsample = nn.ConvTranspose2d(nin, nout, 3, stride=2, padding=1,
                                         output_padding=1)
		self.conv1 = sep_conv(nout*c, nout, 3) if depthwise \
					else nn.Conv2d(nout*c, nout, 3, padding=1)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = sep_conv(nout, nout, 3) if depthwise \
					else nn.Conv2d(nout, nout, 3, padding=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.bn = nn.BatchNorm2d(nout)

	def forward(self, x, skip_feature=None):
		h = self.upsample(x)
		h = torch.cat((skip_feature, h), 1) if self.skip else h
		h = self.relu1(self.conv1(h))
		h = self.relu2(self.conv2(h))
		return self.bn(h)

# Encode image to features		
class Encoder(nn.Module):
	def __init__(self, input_channel):
		super(Encoder, self).__init__()
		self.features = [0 for i in range(5)]
		nin, nout = input_channel, 64
		for i in range(0, 5):
			self.add_module('enc_'+str(i), down_block(nin, nout, bool(i)))
			nin = nout
			nout *= 2
		self.blocks = AttrProxy(self, 'enc_')
		self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

	def forward(self, x):
		h = x.to(device)
		for i in range(0, 5):
			h = self.blocks[i](h)
			self.features[i] = h
		class_feature = self.adaptive_pool(self.features[-1])
		class_feature = torch.reshape(class_feature, (-1, 2*2*1024))
		return self.features[-1], class_feature

# Decode features to image
class Decoder(nn.Module):
	def __init__(self, in_channel, out_channel, skip=True, activation='ReLU'):
		super(Decoder, self).__init__()
		self.skip = skip
		nin, nout = in_channel, 512
		for i in range(0, 4):
			self.add_module('dec_'+str(i), up_block(nin, nout, skip=skip,
								depthwise=True if i != 3 else True))
			nin = nout
			nout //= 2
		self.blocks = AttrProxy(self, 'dec_')
		nout = out_channel
		self.upconv = nn.ConvTranspose2d(nin, nout, 3, stride=2, padding=1,
                                         output_padding=1)
		if activation == 'ReLU':
			self.activation = nn.ReLU(inplace=True)
		elif activation == 'Sigmoid':
			self.activation = nn.Sigmoid()
		elif activation == 'Tanh':
			self.activation = nn.Tanh()
		self.bn5 = nn.BatchNorm2d(nout)

	def forward(self, x, encoder=None):
		h = x
		for i in range(0, 4):
			h = self.blocks[i](h, encoder.features[3-i] if self.skip else None)
		h = self.activation(self.upconv(h))
		return self.bn5(h)

# reconstruction auto-encoder
class AutoEncoder(nn.Module):
	def __init__(self, n_channels):
		super(AutoEncoder, self).__init__()
		self.encoder = Encoder(3)
		self.decoder = Decoder(1024, 3, skip=False, activation='Tanh')

	def forward(self, x):
		features, class_feat = self.encoder(x)
		image = self.decoder(features)
		return image

from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

batch_size = 64
epochs = 10
channels = 3
learning_rate = 1e-3

toTensor = transforms.Compose([transforms.ToTensor()])
toImage = transforms.ToPILImage()
cifar = datasets.CIFAR10(root='./cifar10', download=True,
                         transform=lambda img: toTensor(img.resize((64, 64),
                         Image.ANTIALIAS)))
dataloader = torch.utils.data.DataLoader(cifar, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

autoEncoder = AutoEncoder(channels).to(device)
MSE = torch.nn.MSELoss(reduce=False, size_average=False).to(device)

for epoch in range(epochs):
	optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=learning_rate)
	for i, data in enumerate(dataloader, 0):
		image, labels = data
		reconstruction = autoEncoder(image)
		loss = MSE(image.to(device), reconstruction)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i % 32 == 0:
			print("Iteration: %d, Loss: %f" % (i,
					loss.cpu().detach().numpy()))
			output = toImage(reconstruction.cpu()[0]).convert('RGB')
			output.save('./cifar10_output/%d%d.bmp' % (epoch, i))
	learning_rate *= 0.9
