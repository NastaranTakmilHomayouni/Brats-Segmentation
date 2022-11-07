from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer = 'he_uniform'
def simple_unet_model(IMG_HEIGHT,IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    
    inputs=Input((IMG_HEIGHT,IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))  #(128x128x128x3)
    s=inputs
    
    #Contraction path
    c1=Conv3D(16,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(s) #16 3x3x3 filters
    # results is 128x128x128x16
    c1=Dropout(0.1)(c1) #0.1 of all the 128x128x128 is deleted randomly
    c1=Conv3D(16,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(c1) #16 3x3x3 filters
    # results is 128x128x128x16  -0.1 in each epoch
    p1=MaxPooling3D((2,2,2))((c1)) 
    # results is 64x64x64x16  -0.1 in each epoch
    print('c1 shape is',c1.shape)
    
    c2=Conv3D(32,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(p1) #32 3x3x3 filters
    # results is 64x64x64x32
    c2=Dropout(0.1)(c2) #0.1 of all the 64x64x64 is deleted randomly
    c2=Conv3D(32,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(c2) #32 3x3x3 filters
    # results is 64x64x64x32  -0.1 in each epoch
    p2=MaxPooling3D((2,2,2))((c2)) 
    # results is 32x32x32x32   -0.1 in each epoch
    print('c2 shape is',c2.shape)
    
    c3=Conv3D(64,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(p2) #64 3x3x3 filters
    # results is 32x32x32x64
    c3=Dropout(0.2)(c3) #0.1 of all the 32x32x32 is deleted randomly
    c3=Conv3D(64,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(c3) #64 3x3x3 filters
    # results is 32x32x32x64  -0.1 in each epoch
    p3=MaxPooling3D((2,2,2))((c3)) 
    # results is 16x16x16x64   -0.1 in each epoch
    print('c3 shape is',c3.shape)
    
    c4=Conv3D(128,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(p3) #128 3x3x3 filters
    # results is 16x16x16x128
    c4=Dropout(0.2)(c4) #0.1 of all the 16x16x16 is deleted randomly
    c4=Conv3D(128,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(c4) #128 3x3x3 filters
    # results is 16x16x16x128  -0.1 in each epoch
    p4=MaxPooling3D((2,2,2))((c4)) 
    # results is 8x8x8x128   -0.1 in each epoch
    print('c4 shape is',c4.shape)
    
    c5=Conv3D(256,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(p4) #256 3x3x3 filters
    # results is 8x8x8x256
    c5=Dropout(0.3)(c5) #0.1 of all the 8x8x8 is deleted randomly
    c5=Conv3D(256,(3,3,3),activation='relu',kernel_initializer=kernel_initializer,padding='same')(c5) #256 3x3x3 filters
    # results is 8x8x8x256  -0.1 in each epoch
    print('c5 shape is',c5.shape)
    
    #Expansive path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5) 
    # results is u6=16x16x16x128  c4 is 16x16x16x128
    u6 = concatenate([u6, c4])
    print('u6 shape is',u6.shape)
    # results is u6=16x16x16x256
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6) #128 3x3x3 filters
    c6 = Dropout(0.2)(c6)
    # results is u6=16x16x16x128 #0.2 of all the 16x16x16 is deleted randomly
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    # results is u6=16x16x16x128 -0.1 in each epoch
    print('c6 shape is',c6.shape)
    
    
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6) 
    # results is u7=32x32x32x64  c3 is 32x32x32x64
    u7 = concatenate([u7, c3])
    print('u7 shape is',u7.shape)
    # results is u7=32x32x32x128
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)#64 3x3x3 filters
    c7 = Dropout(0.2)(c7)#results is u7=32x32x32x64 #0.2 of all the 32x32x32 is deleted randomly
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)#64 3x3x3 filters
    # results is u7=32x32x32x64
    print('c7 shape is',c7.shape)
    
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    # results is u8=64x64x64x32 c2 is 64x64x64x32
    u8 = concatenate([u8, c2])
    print('u8 shape is',u8.shape)
    # results is u8=64x64x64x64
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)#32 3x3x3 filters
    c8 = Dropout(0.1)(c8)#results is u8=64x64x64x32 -0.1 in each epoch
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
    # results is u8=64x64x64x32
    print('c8 shape is',c8.shape)
    
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8) 
    # results is u9=128x128x128x16 c1 is 128x128x128x16
    u9 = concatenate([u9, c1])
    print('u9 shape is',u9.shape)
    # results is u9=128x128x128x32
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)#16 3x3x3 filters
    # results is u9=128x128x128x16
    c9 = Dropout(0.1)(c9)
    # results is u9=128x128x128x16 #0.1 of all the 128x128x128 is deleted randomly
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    # results is u9=128x128x128x16 #0.1 of all the 128x128x128 is deleted randomly
    print('c9 shape is',c9.shape)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9) #4 1x1x1 filters for each pixel which label has the highest probability
    
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model


#model = simple_unet_model(128, 128, 128, 3, 4)
#print(model.input_shape)
#print(model.output_shape) 