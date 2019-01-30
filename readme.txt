Neural networks are used to classify bee images and sounds.

How to load each network:

ann = load_image_ann('pck_nets/ImageANN.pck')
convnet = load_image_convnet('pck_nets/ImageConvNet.tfl')
ann = load_audio_ann('pck_nets/AudioANN.pck')
convnet = load_audio_convnet('pck_nets/AudioConvNet.tfl')


