from PIL import Image


im = Image.open('templates/template16.png')
#rotate
angles = [-90,-60,-30,0,30,60,90] #angles de rotation
scales = [1, 0.9,0.8,0.7,0.6,0.5] #coefficient de downsampling

k=1 #numero de l image
for theta in angles :
    for scale in scales :  
        rotate_template = im.rotate(theta)
        rows, cols= rotate_template.size
        rotate_scale_template=rotate_template.resize( (int(scale*cols), int(scale*rows)))
        rotate_scale_template.save('rotate_scale/rotated_scaled'+str(k)+'.png')
        k=k+1
