import torch 
import torch.nn as nn

# สร้าง convolution block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # convolution layer เป็น 3 by 3 convolution ที่มี padding 1
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1) # รับ input จากชั้นที่แล้ว
        self.bn2 = nn.BatchNorm2d(out_c)

        # activation function
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # print(x.shape)
        return x
      
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        

    def forward(self, inputs, skip_connection):
        x = self.up(inputs)
        x = torch.cat([x, skip_connection], axis=1)
        x = self.conv(x)

        return x
    
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)  
        self.e3 = encoder_block(128, 256)  
        self.e4 = encoder_block(256, 512)  

        # Bottleneck
        self.b = conv_block(512, 1024)  

        # Decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1) 
        s3, p3 = self.e3(p2)  
        s4, p4 = self.e4(p3)  

        # bottleneck
        b = self.b(p4)
        # print(s1.shape, s2.shape, s3.shape, s4.shape)
        # print(b.shape)

        # decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs
        

if __name__ == '__main__':
    x = torch.randn((2, 3, 512, 512)) 
    f = build_unet()
    y = f(x)
    print(y.shape)
