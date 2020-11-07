# Note:
To run this, you will need a few different libraries.

# TensorFlowLite:
Install the TensorFlowLite library via the library manager.

# Arducam:
If you are using an Arducam, you will need the Arducam library.
Link: https://github.com/ArduCAM/Arduino
Install it by putting the Arducam folder into your Arduino's Libraries folder. 
Once you install it, you will need to edit the "memorysaver.h" file. 

Edit it to uncomment the model of Arducam that you are using
Example:
Using OV2640 Mini:

- //Step 1: select the hardware platform, only one at a time
- //#define OV2640_MINI_2MP
- //#define OV3640_MINI_3MP
- //#define OV5642_MINI_5MP
- //#define OV5642_MINI_5MP_BIT_ROTATION_FIXED
- #define OV2640_MINI_2MP_PLUS
- //#define OV5642_MINI_5MP_PLUS
- //#define OV5640_MINI_5MP_PLUS

Then, uncomment the correct camera module 
Example:
Using OV2640 Mini:

- //Step 2: Select one of the camera module, only one at a time
- #if (defined(ARDUCAM_SHIELD_REVC) || defined(ARDUCAM_SHIELD_V2))
-	//#define OV7660_CAM
-	//#define OV7725_CAM
-	//#define OV7670_CAM
-	//#define OV7675_CAM
- #define OV2640_CAM
-	//#define OV3640_CAM
-	//#define OV5642_CAM
-	//#define OV5640_CAM
- #endif

# JPEGDecoder:
Install the JPEGDecoder library via the library manager. 

