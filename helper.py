import torch
import matplotlib.pyplot as plt
def show_im(im,normalized,mean,std):
    """
    polt the input image

    :param im: input image of type pytorch tensore (C x W X H)
    :param normalized: Boolean value. if the input has been normalized by mean and std
    :param mean: the mean value used to normalize the input there is one value for each channel [m_ch1, m_ch2, m_ch3]
    :param std: the std value used to normalize the input there is one value for each channel [sd_ch1, sd_ch2, sd_ch3]
    :return: no return
    """
    if normalized:
        # reshape the mean and std to match the input
        std = torch.Tensor(std).double().view(3,1,1)
        mn = torch.Tensor(mean).double().view(3,1,1)
        # denormalize the input to get the original value
        im = im * std + mn
    im = im.numpy() # convert to numpy
    im = im.transpose((1,2,0)) # transpose from (C X W X H) to (W X H X C)
    plt.imshow(im) # plot the image