def run_server():
    import io
    from flask import Flask, render_template, request, send_file,redirect, url_for
    from flask_uploads import UploadSet, configure_uploads, IMAGES
    #from model import evaluate
    from checkpoint_image_caption import evaluate
    import os
    import random
    import string
    import numpy as np
    import cv2
    import tensorflow as tf
    import time

    app = Flask(__name__)
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        global files
        if request.method == 'POST' and 'photo' in request.files:
            filename_raw = request.files['photo'].filename
            filestr = request.files['photo'].read()
            npimg = np.fromstring(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            save=cv2.imwrite(r'/Users/rishithareddy/Desktop/ImageCaptioning/doc/'+filename_raw, img)
            path='/Users/rishithareddy/Desktop/ImageCaptioning/doc/'+filename_raw
            test_res,attention = evaluate(path)
            result='Prediction Caption : '+ ' '.join(test_res)
            return render_template('display.html',caption=result,filename=filename_raw,img=img,path=path)
        return render_template('upload.html')


    
    app.run(host = '0.0.0.0',debug=True)



if __name__ == "__main__":
    run_server()