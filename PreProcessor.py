from scipy.fftpack import dct
import numpy as np

from cv2 import imread, imencode, imdecode, cvtColor, IMREAD_COLOR, IMWRITE_JPEG_QUALITY, COLOR_BGR2YCrCb


def get_lead_digit(number):
    # if number == 0: # if the number is 0, then return a value of 0 for count
    #    return None # change to 0 or none. value range in YCbCr actually doesn't include zero`
    for s in str(number):
        if s != '0' and s.isdigit():  # if the first encountered digit is not 0 and also is a digit, return it
            return int(s)


def get_leading_digits(vect):
    # M is the number of lead digits we will collect
    M = vect.shape[0]
    lead_digits = np.empty(vect.shape)  # vector of leading digits (including zero values)

    for k in range(M):
        num = np.abs(vect[k])  # absolute value each number

        if num == 0:  # skips 0 values
            continue

        lead_digits[k] = get_lead_digit(num)  # get the lead digit

    return lead_digits


def get_lead_hist(unr_block):
    lead_digits = get_leading_digits(unr_block)  # counts the leading digits of all 3 YCbCr Channels
    counts = np.histogram(lead_digits, bins=9, range=(1., 9.))[0]  # put all leads into 9 bins of values 1-9

    return counts


class PreProcessor:
    zigzag_index = (
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63,
    )

    def __init__(self, is_dataset=True, JPEG_QUALITY=None):
        self._X = np.empty(0)
        self.myimgbytes = None
        self.is_dataset = is_dataset
        self.imshape = None
        return

    def get_shape(self):
        return self.imshape
        
    def get_datapoints(self):
        return self._X.shape

    def get_data(self):
        return self._X

    def get_cols(self):
        # creating the column labels for csv exportation
        cols = ['coeff' + str(i) for i in range(64 * 1)] + ['hist' + str(j) for j in range(9)]
        return cols


    def unravel_block(self, block):
        unraveled = np.zeros(64)  # unraveled vector
        i = 0

        for row in block:
            for e in row:
                unraveled[self.zigzag_index[i]] = e  # assign element to the ith index value of in zigzag_index
                i += 1  # increment for each visited element in the array

        return unraveled

    def block_dct(self, block):

        dctii = dct(block, norm='ortho')  # dct coeff block Y channel
        unr_dct = self.unravel_block(dctii)  # unravel Y dct

        return unr_dct

    # takes 8x8 convolutions of each image (the same size as jpeg compression)
    # applies DCT to the 8x8 block before unravelling in a zig-zag like pattern.
    # finally, counts the trailing digits of each channel of YCbCr pixel value.
    def build_features(self, img):  # , compressed=0):
        # create image object from RGB and convert it to YCbCr space
        pimg = cvtColor(img, COLOR_BGR2YCrCb)

        # make it a manipulable matrix
        pimg_arr = np.array(pimg)
        dims = pimg_arr.shape
        stride = 8  # 8x8 stride/blocks for each datapoint

        for i in range(0, (dims[0] - dims[0] % stride), stride):  # moves across columns
            for j in range(0, (dims[1] - dims[1] % stride), stride):  # moves accross rows

                block0 = pimg_arr[i:i + 8, j:j + 8, 0]  # image block Y
                unr_block = img[i:i + 8, j:j + 8].ravel()

                # 64 count unr dct of y-channel and 9 bin histogram of all 3 channel values
                x = np.append(self.block_dct(block0), get_lead_hist(unr_block))

                yield x

    def process_datapoints(self, myimgbytes):
        # this function has to be called in order to initialize JPEG quality and generate dataset.
        # default dataset can be found here: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
        self.myimgbytes = myimgbytes

        X = np.empty(0)
        indx = 0  # this is the image number

        #print(f'processing image')

        image = imdecode(myimgbytes, IMREAD_COLOR)#imread(myimg, IMREAD_COLOR)
        self.myimg = image
        self.imshape = image.shape

        # get datapoint generator for this particular image (yielded from fcn) as never compressed
        # make it an array, and unravel/append it to X. can reshape at the end.
        X = np.append(X, np.array([i for i in self.build_features(image)], dtype=object))

        self._X = X
        self.reshape_data()
        #print("Processing Completed")

    def reshape_data(self):
        # reshape into m samples by 73 values: (73 features, a label and img number)
        self._X = self._X.reshape(self._X.shape[0] // ((64 * 1) + 9), ((64 * 1) + 9))  # no labels

        return
