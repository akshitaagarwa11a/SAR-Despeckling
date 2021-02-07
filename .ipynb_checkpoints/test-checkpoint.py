from model import UNet

path = "./checkpoints/checkpoint_SAR.pt"
model_test = UNet()
model_test.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

for filename in os.listdir('./testimages'):
    if not filename.startswith('.'):
        image = Image.open('./testimages/' + filename).convert("LA")
        mean, std = np.mean(image), np.std(image)
        x=9
        data_transforms = transforms.Compose([
                                             transforms.RandomResizedCrop(2**x),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5)
        ])
        img = data_transforms(image)[0]
        img = torch.reshape(img, (1, 1, 2**x, 2**x))
        image_log = lintolog(img)
        img_out = logtolin(model_test(image_log))
        plt.imsave('./output/'+filename, img_out[0][0], cmap='gray')
        fig2 = plt.figure(figsize = (10,10)) # create a 5 x 5 figure
        ax2 = fig2.add_subplot(121)
        ax2.imshow(img_out[0][0], interpolation='none', cmap='gray')
        ax2 = fig2.add_subplot(122)
        ax2.imshow(img[0][0], interpolation='none', cmap='gray')
        ax2.set_title(filename)