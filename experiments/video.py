import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
import utils
from utils import StyleLoader

def run_demo(args, mirror=False):
    #  style_model = Net(ngf=args.ngf)
    #  style_model.load_state_dict(torch.load(args.model))

    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy() # We can't mutate while iterating

    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]

    style_model = Net(ngf=args.ngf)
    style_model.load_state_dict(model_dict, False)

    style_model.eval()
    if args.cuda:
            style_loader = StyleLoader(args.style_folder, args.style_size)
            style_model.cuda()
    else:
            style_loader = StyleLoader(args.style_folder, args.style_size, False)

    # Define the codec and create VideoWriter object
    height =  args.demo_size
    width = int(4.0/3*args.demo_size)
    swidth = int(width/4)
    sheight = int(height/4)


    #lxy
    if args.video == '':
        videofile = '/home/xyliu/Videos/sports/dance.mp4'
    else:
        videofile = args.video

    print('handle video\'s path is', videofile)
    cam = cv2.VideoCapture(videofile)

    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ret_val, img = cam.read()
    i = 1

    ## output name
    if args.output == '':
        dirname = os.path.dirname(videofile)
        basename = os.path.basename(videofile)
        output_name = os.path.join(dirname, 'tra_'+basename)
    else:
        output_name = args.output

    if args.record:
            fshape = img.shape
            fheight, fwidth = fshape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #  out = cv2.VideoWriter('new_output.mp4', fourcc, 20.0, (2*width, height))
            out = cv2.VideoWriter(output_name, fourcc, 20.0, (fwidth*2, fheight))

    cam.set(3, width)
    cam.set(4, height)
    key = 0
    idx = 0
    while(cam.isOpened()) and ret_val == True and i < video_length:
            # read frame
            idx += 1
            ret_val, img = cam.read()
            i += 1
            if mirror:
                    img = cv2.flip(img, 1)
            cimg = img.copy()
            img = np.array(img).transpose(2, 0, 1)

            # changing style
            if idx%20 == 1:
                    style_v = style_loader.get(int(idx/20))
                    style_v = Variable(style_v.data)
                    #  print('style change to ', style_v)
                    style_model.setTarget(style_v)

            img=torch.from_numpy(img).unsqueeze(0).float()
            if args.cuda:
                    img=img.cuda()

            img = Variable(img)

            ## handle image
            img = style_model(img)

            if args.cuda:
                    simg = style_v.cpu().data[0].numpy()
                    img = img.cpu().clamp(0, 255).data[0].numpy()
            else:
                    simg = style_v.data().numpy()
                    img = img.clamp(0, 255).data[0].numpy()
            img = img.transpose(1, 2, 0).astype('uint8')
            simg = simg.transpose(1, 2, 0).astype('uint8')

            # display
            simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
            cimg[0:sheight,0:swidth,:]=simg
            img = np.concatenate((cimg,img),axis=1)
            cv2.imshow('MSG Demo', img)
            #cv2.imwrite('stylized/%i.jpg'%idx,img)
            key = cv2.waitKey(1)
            if args.record:
                    out.write(img)
            if key == 27:
                    break
    cam.release()
    if args.record:
            out.release()
    cv2.destroyAllWindows()
    print('video save in ', output_name)

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)


if __name__ == '__main__':
	main()
