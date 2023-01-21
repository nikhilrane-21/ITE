import os
import json
import pymongo
from pymongo import MongoClient
import cv2
import re
import ctypes
from PIL import Image
import img2pdf
import pandas as pd
import pytesseract
import math
import numpy as np
from typing import Tuple, Union
from deskew import determine_skew
import cv2
# import argparse
from pdf2image import convert_from_path
# from pytesseract import image_to_string, image_to_data
import streamlit as st
from streamlit_drawable_canvas import st_canvas

global objects
st.set_page_config(
    page_title='Welcome to Text Extractor from Invoices',
    page_icon=':robot:',
    layout='wide'
)
st.title("Invoice Text Extractor")

# image_col, text_col = st.columns(2)
file = st.file_uploader("Upload a invoice", type=["jpg", "jpeg", "png", "pdf", "tiff"])

if file is not None:
    file_details = {"FileName":file.name,"FileType":file.type}
    with open(os.path.join("tempDir",file.name),"wb") as f:
      f.write(file.getbuffer())

# with image_col:
# Specify canvas parameters in application
drawing_mode = st.selectbox(
    "Drawing tool:", ("rect","transform")
)
dim = cv2.imread("page0.jpg")
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
    stroke_width=1,
    stroke_color="#000000",
    background_image=Image.open('page0.jpg') if file else None,
    height= dim.shape[0],
    width= dim.shape[1],
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])        # need to convert obj to str because PyArrow


datetext = ""
invnotext = ""
billtext = ""
regexbilltext = ""
buyertext = ""
sellertext = ""


def isValid(total_amount):
    total_amount=total_amount.strip()
    if(len(total_amount)==0):
        print("tr")
        return False
    elif(len(total_amount)>0):
        for ch in total_amount:
            if(ch!="." and ch!="," and ch.isdigit()!=True):
                return False
    return True


def match_last(orig_string, regex):

    re_lookahead = re.compile(regex)
    match = None
    for match in re_lookahead.finditer(orig_string):
        pass

    if match:
        re_complete = re.compile(regex)
        return re_complete.match(orig_string, match.start())
    return match


def method3():
    global regexbilltext

    # images = convert_from_path(file.name, poppler_path="C:/poppler-0.68.0/bin")
    keyword = matched_doc

    # Getting the mongo db document of the matched template
    matched_invoice = db.invoices.find({"keyword": keyword.strip()})

    start = 0
    count = 0
    for invoice in matched_invoice:
        count += 1
        start = invoice["match_start"]  # Starting word in sandwich technique that will be searched for
        end = invoice["match_end"]  # Ending word in sandwich technique that will be searched for

    # if no starting word can be found
    if (start == "-1"):
        count = 0
    flag = -1

    # Finally searching for the matched starting and ending keyword and returning the amount stored b/w them
    if count == 1:
        text = fullinvoicetext

        # Searching for a string that has start and end and any string stored in between in it in the text of the pdf
        regex = r"(?s)(?<="+start+r")(.*?)(?="+end+r")"
        start_end = re.search(regex, text)
        # start_end = match_last(text, regex)
        # if (start_end == None):
        #     pass

        # If the start and end is found
        if (start_end != None):
            flag = 0
            text = start_end.group()

            # regex to get the amount stored between the start and end words
            amount_regex = "[0-9.,]+\n"
            total_amount_3 = match_last(text, amount_regex)
            total_amount_3 = total_amount_3.group()
            total_amount_3 = total_amount_3.strip()

            with open('out.txt', 'w') as f:
                f.write(total_amount_3)

        # If no amount is found, write false to indicate in the temporary output file
        if (flag == -1):
            with open('out.txt', 'w') as f:
                f.write("false, no amount")
    else:
        with open('out.txt', 'w') as f:
            f.write("-1, write to file fail")
        print("-1, write to file fail")

    outfile = open("out.txt")
    total_amount_3 = outfile.read()
    regexbilltext = total_amount_3
    outfile.close()
    return total_amount_3


# function to rotate and remove the skew from the image
def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
            abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
             abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


# function  to get the starting parameter and ending paramter for part 3
def getstartend(total_amount, text):
    lines = text.split("\n")

    # remove the empty lines
    non_empty_lines = [line for line in lines if line.strip() != ""]
    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    text = string_without_empty_lines  # This conatins no empty lines
    with open('textforstartend.txt', 'w') as f:
        f.write(text)

    total_amount = total_amount.strip()

    # regex to match the first word in the line where total amount occurs and the first of the next line of total amount.
    # These are the start and the end parameters respectively
    isAvailable= re.search(f"{total_amount}", text)
    if isAvailable != None:
        start_end = re.search('\n.*?' + f"\s{total_amount}" + '.*?\n\w+', text)

        if (start_end != None):
            start = re.search(r'[a-zA-Z]+', start_end.group())
            start = start.group()
            end = start_end.group().split("\n")[-1].strip()
            return start, end
        else:
            # If no regex matches
            return "-1", "-1"
    else:
        return "-1", "-1"

# function to get the desired fields when a template has been matched for the pdf
def getinf(item):

    global datetext
    global invnotext
    global billtext
    global buyertext
    global sellertext

    # Cropping and saving the rectangle that contains the keyword
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["keyword_cordinates"]["x1"] + shiftx, item["keyword_cordinates"]["y1"] + shifty, item["keyword_cordinates"]["x2"] + shiftw, item["keyword_cordinates"]["y2"] + shifth))
    img1.save('img2.png')
    img1.close()

    # Cropping and saving the rectangle that contains the Date of Invoice
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["Date"]["x1"] + shiftx, item["Date"]["y1"] + shifty, item["Date"]["x2"] + shiftw, item["Date"]["y2"] + shifth))
    img1.save('date.png')
    img1.close()
    # reading text from the cropped image to get the Date of Invoice
    datetext = str(pytesseract.image_to_string(Image.open(r"date.png"), lang='eng'))

    # reading the Invoice No after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["Invoice_No"]["x1"] + shiftx, item["Invoice_No"]["y1"] + shifty, item["Invoice_No"]["x2"] + shiftw, item["Invoice_No"]["y2"] + shifth))
    img1.save('invno.png')
    img1.close()
    invnotext = str(pytesseract.image_to_string(Image.open(r"invno.png"), lang='eng'))

    # reading the Total bill after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["Total Bill"]["x1"] + shiftx, item["Total Bill"]["y1"] + shifty, item["Total Bill"]["x2"] + shiftw, item["Total Bill"]["y2"] + shifth))
    img1.save('bill.png')
    img1.close()
    billtext = str(pytesseract.image_to_string(Image.open(r"bill.png"), lang='eng'))
    total_amount = billtext

    # reading the Buyer Address after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["Buyer"]["x1"] + shiftx, item["Buyer"]["y1"] + shifty, item["Buyer"]["x2"] + shiftw, item["Buyer"]["y2"] + shifth))
    img1.save('buyer.png')
    img1.close()
    buyertext = str(pytesseract.image_to_string(Image.open(r"buyer.png"), lang='eng'))


    # reading the Seller Address  after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((item["Seller"]["x1"] + shiftx, item["Seller"]["y1"] + shifty, item["Seller"]["x2"] + shiftw, item["Seller"]["y2"] + shifth))
    img1.save('seller.png')
    img1.close()
    sellertext = str(pytesseract.image_to_string(Image.open(r"seller.png"), lang='eng'))

    return total_amount


# Connection to db

connection = pymongo.MongoClient('mongodb+srv://nikhil:nikhil@atlascluster.7o742.mongodb.net/?retryWrites=true&w=majority')
db = connection.helloworld

# Intializing Some variables for part3
ref_point = []
total_amount = ""
crop = False
onboard = 0
flag = 0
shiftx = 0
shifty = 0
shiftw = 0
shifth = 0

if file is not None:
    # converting pdf into images for every page of the pdf
    images = convert_from_path("./tempDir/"+file.name)

    # Extracting the image of each pages from the pdf
    no_of_pages = len(images)
    for i in range(len(images)):
        images[i].save('page' + str(i) + '.jpg', 'JPEG')

    found = 0  # variable to check if a matching template is found
    image = cv2.imread('page0.jpg')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)  # determine the skew angle prsent in the original image
    rotated = rotate(image, angle, (0, 0, 0))  # cancelling the skew in the original image and rorated is the new image after cancelling the skew in the original image
    cv2.imwrite('page0.jpg', rotated)
    matched_doc = ""  # to store the temaplate that has match with the template

    # extracting the text of the page1
    text = str(pytesseract.image_to_string(Image.open(r"page0.jpg"), lang='eng'))
    fullinvoicetext = text
    # check if a matching template exists with teh same keyword and keyword bounding boxes
    for document in db.invoices.find():
        img1 = Image.open("page0.jpg")
        img3 = img1.crop((document["keyword_cordinates"]["x1"], document["keyword_cordinates"]["y1"], document["keyword_cordinates"]["x2"], document["keyword_cordinates"]["y2"]))
        img3.save('img3.png')
        img3.close()
        image = Image.open('img3.png')
        image.close()
        text = str(pytesseract.image_to_string(Image.open(r"img3.png"), lang='eng'))
        text = text.strip()
        foundregex = re.search(r'[a-zA-Z]+', text)
        if (foundregex != None):
            text = foundregex.group()
        key = document["keyword"].strip()
        if (text == key):
            found = 1
            total_amount = getinf(document)
            matched_doc = document["keyword"].strip()
            if (no_of_pages != document["no_of_pages"]):
                print("-1")
                flag = -1
                break

    # pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe" # to make tesseract work, put in the exact path of tesseract

    # matching every word of the pdf to every keyword to find the shift and confirming by matching with seller key
    if (found == 0):
        img = Image.open('page0.jpg')
        data = pytesseract.image_to_data(img, output_type='dict')
        boxes = len(data['level'])
        for document in db.invoices.find():
            for i in range(boxes):
                if data['text'][i].strip() != ''.strip():
                    key_to_match = data['text'][i]
                    foundregex = re.search(r'[a-zA-Z]+', key_to_match)

                    # Checking only for valid strings
                    if (foundregex != None):
                        key_to_match = foundregex.group()

                        # If keyword matches with a word in the pdf, get the x, y shift of teh keyword coordinates stored
                    if ((key_to_match) == document["keyword"]):

                        shiftx=data["left"][i]-document["keyword_cordinates"]["x1"]
                        shifty=data["top"][i]-document["keyword_cordinates"]["y1"]
                        shiftw=data["left"][i]-document["keyword_cordinates"]["x1"]
                        shifth=data["top"][i]-document["keyword_cordinates"]["y1"]
                        img1 = Image.open("page0.jpg")
                        img1 = img1.crop((document["Seller"]["x1"] + shiftx, document["Seller"]
                        ["y1"] + shifty, document["Seller"]["x2"] + shiftw, document["Seller"]["y2"] + shifth))
                        img1.save('img2.png')
                        img1.close()
                        found = 1
                        break

            if (found == 1):
                matched_doc = document["keyword"].strip()
                total_amount = getinf(document)
                break

    if found == 0:
        onboard = 1
        st.header("No Template found for this Invoice; \nPlease Start the onboarding\n")
        # if st.button("Onboard"):
        st.text("Select boxes on the displayed image in the following order: \n1.Keyword\n2.Date of Invoice\n3.Invoice No.\n4.Buyer Details\n5.Seller Details")
        curr = 1

        # If valid selection is made by user while dragging the rectangle, i.e, atleast one rectangle has been selected successfully
        if (objects.shape[0]==6):
            img1 = Image.open("page0.jpg")
            # cropping and saving only the rectangle portion of the image from where we have to extract the text
            img3 = img1.crop((objects["left"][0], objects["top"][0], objects["left"][0] + objects["width"][0],
                              objects["top"][0] + objects["height"][0]))
            img3.save('img2.png')
            img3.close()

            image = Image.open('img2.png')
            image.close()

            # text variable contains the text in the bounding box selected
            text = str(pytesseract.image_to_string(
                Image.open(r"img2.png"), lang='eng'))

            # getting the x and y coordinates of the keyword  from the pdf, for this the text read from the bounding box is matxhed with edvery word
            # of the pdf and when the word matches we store its x, y coordinates as the keyword's coordinates. This is done to ensure that we dont consider the extra region of the image that ahs not
            # text , we want a tight bound to x y coordinates for keyword
            text = text.strip()
            foundregex = re.search(r'[a-zA-Z]+', text)
            if (foundregex != None):
                text = foundregex.group()

            # converting the whole pdf to text to match its every word
            myimg = Image.open('page0.jpg')
            data = pytesseract.image_to_data(myimg, output_type='dict')
            boxes = len(data['level'])

            for i in range(boxes):
                key_to_match = data['text'][i].strip()
                if (key_to_match != ""):
                    foundregex = re.search(r'[a-zA-Z]+', key_to_match)
                    if (foundregex != None):
                        key_to_match = foundregex.group()

                    if key_to_match.strip() == text.strip():
                        break

            # inserting the new template in the db
            db.invoices.insert_one({"keyword": text})
            keyword = text
            db.invoices.update_one({"keyword": keyword}, {"$set": {"no_of_pages": no_of_pages}})
            left = np.int64(objects["left"][0])
            left = left.item()
            top = np.int64(objects["top"][0])
            top = top.item()
            right = np.int64(objects["left"][0] + objects["width"][0])
            right = right.item()
            bottom = np.int64(objects["top"][0] + objects["height"][0])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {"$set": {"keyword_cordinates": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})

            # extracting the date of invoice from the bounding box selected for it and storing its keyword
            # coordinates in the database
            img3 = img1.crop((objects["left"][1], objects["top"][1], objects["left"][1] + objects["width"][1], objects["top"][1] + objects["height"][1]))
            left = np.int64(objects["left"][1])
            left = left.item()
            top = np.int64(objects["top"][1])
            top = top.item()
            right = np.int64(objects["left"][1] + objects["width"][1])
            right = right.item()
            bottom = np.int64(objects["top"][1] + objects["height"][1])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {"$set": {"Date": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})
            img3.save('date.png')
            img3.close()
            image = Image.open('date.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"date.png"), lang='eng'))
            datetext = text.strip()

            # extracting the Invoice No. from the bounding box selected for it and storing its keyword
            # coordinates in the database
            img3 = img1.crop((objects["left"][2], objects["top"][2], objects["left"][2] + objects["width"][2], objects["top"][2] + objects["height"][2]))
            left = np.int64(objects["left"][2])
            left = left.item()
            top = np.int64(objects["top"][2])
            top = top.item()
            right = np.int64(objects["left"][2] + objects["width"][2])
            right = right.item()
            bottom = np.int64(objects["top"][2] + objects["height"][2])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {
                "$set": {"Invoice_No": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})
            img3.save('invno.png')
            img3.close()
            image = Image.open('invno.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"invno.png"), lang='eng'))
            invnotext = text.strip()

            # extracting the Total Bill from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop((objects["left"][3], objects["top"][3], objects["left"][3] + objects["width"][3], objects["top"][3] + objects["height"][3]))
            left = np.int64(objects["left"][3])
            left = left.item()
            top = np.int64(objects["top"][3])
            top = top.item()
            right = np.int64(objects["left"][3] + objects["width"][3])
            right = right.item()
            bottom = np.int64(objects["top"][3] + objects["height"][3])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {
                "$set": {"Total Bill": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})
            img3.save('bill.png')
            img3.close()
            image = Image.open('bill.png')
            image.close()
            billtext = str(pytesseract.image_to_string(Image.open(r"bill.png"), lang='eng'))
            billtext = billtext.strip()
            total_amount = billtext

            # Storing the word before and after the total bill for part 3
            wholetext = str(pytesseract.image_to_string(
                Image.open(r"page0.jpg"), lang='eng'))
            wholetext = wholetext.strip()
            match_start, match_end = getstartend(total_amount, wholetext)

            db.invoices.update_one({"keyword": keyword}, {
                "$set": {"match_start": match_start}})
            db.invoices.update_one({"keyword": keyword}, {
                "$set": {"match_end": match_end}})

            # extracting the Buyer Address from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop((objects["left"][4], objects["top"][4], objects["left"][4] + objects["width"][4], objects["top"][4] + objects["height"][4]))
            left = np.int64(objects["left"][4])
            left = left.item()
            top = np.int64(objects["top"][4])
            top = top.item()
            right = np.int64(objects["left"][4] + objects["width"][4])
            right = right.item()
            bottom = np.int64(objects["top"][4] + objects["height"][4])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {
                "$set": {
                    "Buyer": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})
            img3.save('buyer.png')
            img3.close()
            image = Image.open('buyer.png')
            image.close()
            buyertext = str(pytesseract.image_to_string(
                Image.open(r"buyer.png"), lang='eng'))
            buyertext = buyertext.strip()

            # extracting the Seller Address from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop((objects["left"][5], objects["top"][5], objects["left"][5] + objects["width"][5], objects["top"][5] + objects["height"][5]))
            left = np.int64(objects["left"][5])
            left = left.item()
            top = np.int64(objects["top"][5])
            top = top.item()
            right = np.int64(objects["left"][5] + objects["width"][5])
            right = right.item()
            bottom = np.int64(objects["top"][5] + objects["height"][5])
            bottom = bottom.item()
            db.invoices.update_one({"keyword": keyword}, {
                "$set": {"Seller": {"x1": left, "y1": top, "x2": right, "y2": bottom}}})
            img3.save('seller.png')
            img3.close()
            image = Image.open('seller.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"seller.png"), lang='eng'))
            sellertext = text.strip()
            sellerkey = sellertext.split("\n")[0]
            db.invoices.update_one({"keyword": keyword}, {"$set": {"seller_key": sellerkey}})



    total_amount = total_amount.strip()
    total_amount_3 = method3()


    st.header("Extracted Data\n")
    st.write({"Invoice Date: ": datetext,
         "Invoice Number: ": invnotext,
         "Invoice Amount: ": billtext,
         "Invoice Buyer: ": buyertext,
         "Invoice Seller: ": sellertext,
         })
    st.write("Regex Amount: ", regexbilltext)
    if st.button("WrongData? Onboard Manually"):
        st.text(
            "Select boxes in the order:\n1.Keyword\n2.Date of Invoice\n3.Invoice No.\n4.Buyer Details\n5.Seller Details")
        image = cv2.imread("page0.jpg")
        curr = 1

        # If valid selection is made by user while dragging the rectangle, i.e, atleast one rectangle has been selected successfully
        if len(ref_point) >= 2:
            img1 = Image.open("page0.jpg")

            # extracting the date of invoice from the bounding box selected for it and storing its keyword
            # coordinates in the database
            img3 = img1.crop(
                (ref_point[2][0], ref_point[2][1], ref_point[3][0], ref_point[3][1]))
            img3.save('date.png')
            img3.close()
            image = Image.open('date.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"date.png"), lang='eng'))
            datetext = text.strip()

            # extracting the Invoice No. from the bounding box selected for it and storing its keyword
            # coordinates in the database
            img3 = img1.crop((ref_point[4][0], ref_point[4][1], ref_point[5][0], ref_point[5][1]))
            img3.save('invno.png')
            img3.close()
            image = Image.open('invno.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"invno.png"), lang='eng'))
            invnotext = text.strip()

            # extracting the Total Bill from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop(
                (ref_point[6][0], ref_point[6][1], ref_point[7][0], ref_point[7][1]))
            img3.save('bill.png')
            img3.close()
            image = Image.open('bill.png')
            image.close()
            billtext = str(pytesseract.image_to_string(Image.open(r"bill.png"), lang='eng'))
            billtext = billtext.strip()
            total_amount = billtext

            # extracting the Buyer Address from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop(
                (ref_point[8][0], ref_point[8][1], ref_point[9][0], ref_point[9][1]))
            img3.save('buyer.png')
            img3.close()
            image = Image.open('buyer.png')
            image.close()
            buyertext = str(pytesseract.image_to_string(
                Image.open(r"buyer.png"), lang='eng'))
            buyertext = buyertext.strip()

            # extracting the Seller Address from the bounding box selected for it and storing its keyword coordinates in the database
            img3 = img1.crop(
                (ref_point[10][0], ref_point[10][1], ref_point[11][0], ref_point[11][1]))
            img3.save('seller.png')
            img3.close()
            image = Image.open('seller.png')
            image.close()
            text = str(pytesseract.image_to_string(
                Image.open(r"seller.png"), lang='eng'))
            sellertext = text.strip()
            sellerkey = sellertext.split("\n")[0]

        cv2.destroyAllWindows()
        st.header("Manually Extracted Data\n")
        st.write({"Invoice Date: ": datetext,
             "Invoice Number: ": invnotext,
             "Invoice Amount: ": billtext,
             "Invoice Buyer: ": buyertext,
             "Invoice Seller: ": sellertext,
             })
