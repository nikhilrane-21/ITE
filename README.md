## Data extractor for PDF invoices 


INSTRUCTIONS TO RUN:

USAGE:

Python version: Python 3.10.7

Download tesseract exe from https://github.com/UB-Mannheim/tesseract/wiki.

Direct link for the tesseract binary : https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.2.0.20220712.exe

Install this exe in `C:\Program Files\Tesseract-OCR`

OR

To make tesseract work, put in the exact path of tesseract in `pytesseract.tesseract_cmd` in file `script_withoutdb.py` line no. 275.

Download `'cmake'` from https://cmake.org/download/  

Add the `C:\Program Files\CMake\bin` path to systems' environment variables path.


Download Poppler binary : https://blog.alivate.com.au/poppler-windows/

Extract the zip into `C:\Program Files\`

Add the bin folder path to systems' environment variables path. example path : `C:\Program Files\poppler-x.x.x\bin`



Create the Virtual environment : `python -m venv env`

Enable Environment : `.\env\Scripts\activate`

Install all the required dependencies : `pip install -r requirements.txt`




The algorithm uses 3 methods:

**Bounding Boxes:** In this bounding boxes are selected for desired fields that re keyword, Date of Invoice, Invoice No. Total Bill Amount, Buyer Address and Seller Address in the same order. To select a bounding box , click on the left most corner and then hold the mouse button and drag till the rightmost bottom corner. Then release the mouse button.The x and y coordinates for the topleft and bottomright corners are stored in rhe database mapped with their keywords. The next time an invoice is fed into the system, it searches for the matching keyword and if it finds it then it uses the already stored x y coordinates to extract the r5quired fields. if the system is unable to match the keyword it, we have to onboard a new template for that invoice.

   The keyword is matched in the following way, 
   i. First for every document in the database we check that if in the rectangle bounded by the keyword coordinates eact same keyword is present, if yes the thenkeyword is found.
   ii. Otherwise every word of the pdf is checked against every keyword present in the database, if a word of the pdf matches with any of the leywords the the first line of the seller address is matched to confirm for the template. If found the that tempalte is considered, otherwise onboarding mode is set on.

Code for this is contained in part1.py

**Invoice Net Training Data:** In this method we use pretrained models to predict the total amount from the bill.

**NLP Method:** In this method, part 1 the document matched in part 1 is used. This first extracts the text of the whole pdf and then tries to match the word just after and before the total bill, if tehse words are matched in the template and the document tehen the text between that is returned as the total amount.Code for this is contained in part3.py


To run the code:

1. Use the command:  
   `python final.py --pdf pdfs/invoice.pdf`
2. First part1 is run
2. A window will open up, if the invoice template has not been added before. Select the desired rectangle by dragging with help of   mouse and then press c on keyboard
3. The window will close and the text will be showed on the terminal.
4. And if the template was added earlier and a document can be matched, the extracted fields are shown inthe terminal.
5. Then after part 1 is run and the onboarding mode is not on, the total amount is extracted by method 2 and if the total amount obtained by part1 and part2 are same. This total amount is returned and the program stops.
6. But if the total amount from part1 and part2 dont match and total amount of both part1 and aprt2 are valid then total amount extracted from par2 is returned.
7. If the total amount is not valid in part1 and part2 both, then method 3 is executed and if the total amount extracted form part 3 id valid it is returned else -1 is returned.

If the pretrained model is not present, first preoare and train the model for part 2.
<br>Use the following commands for this:
<br>1. `python InvoiceNet/prepare_data.py --data_dir InvoiceNet/train_data4/`
<br>2. `python InvoiceNet/train.py --field total_amount --batch_size 2`

Refer to InvoiceNet documention for more details about part2.


