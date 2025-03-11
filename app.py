from io import BytesIO
import logging
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
from Code import CNN_Model
from flask_cors import CORS
import cv2
import numpy as np
from skimage.filters import threshold_otsu
import base64

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)
cors = CORS(app)

# Initialize the CNN model and generate plots
object = CNN_Model()
#make model ready to use
object.TrainModel()
def highlight_tumor(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    threshold = 0.5
    _, mask = cv2.threshold(grayscale_image, threshold * 255, 255, cv2.THRESH_BINARY)

    overlay = np.zeros_like(resized_image)
    overlay[:, :, 2] = np.where(mask > 0, 255, 0)

    result = cv2.addWeighted(resized_image, 1, overlay, 0.5, 0)
    result = cv2.resize(result, (image.shape[1], image.shape[0]))

    return result
# img=object.fig2img("./uploads/test.jpg")
# print(img)
# plot1, plot2, plot3, plot4 = object.TrainModel()
# plot1_img = secure_filename("plot1.png")
# plot2_img = secure_filename("plot2.png")
# plot3_img = secure_filename("plot3.png")
# plot4_img = secure_filename("plot4.png")
# plot1.save(os.path.join('uploads', plot1_img))
# plot2.save(os.path.join('uploads', plot2_img))
# plot3.save(os.path.join('uploads', plot3_img))
# plot4.save(os.path.join('uploads', plot4_img))
print("CNN model and plots initialized")
# logging.info("CNN model and plots initialized")


@app.route('/')
def index():
    return 'Flask server running...'


# @app.route('/api/classify_image', methods=['POST'])
# def classify_image():
#     logging.info("Received a POST request to classify_image endpoint")
#     # Check if the POST request has the file part
#     if 'file' not in request.files:
#         logging.error("No file part in the POST request")
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     # Check if the file is one of the allowed file types
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join('uploads', filename))
#         output = object.classifyImage(os.path.join('uploads', filename))
#         os.remove(os.path.join('uploads', filename))
#         logging.info("Image classification completed successfully")
#         return jsonify({'output': output})
#     else:
#         logging.error("File type not allowed")
#         return jsonify({'error': 'File type not allowed'})
def image_to_base64(image):
    # Convert the image ndarray to PIL Image format
    pil_image = Image.fromarray(image)

    # Create a BytesIO object to store the image data
    image_buffer = BytesIO()

    # Save the PIL Image as PNG to the BytesIO object
    pil_image.save(image_buffer, format='PNG')

    # Encode the image data in base64
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    return image_base64
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # logging.info("Received a POST request to upload_image endpoint")
    print("Received a POST request to upload_image endpoint")
    try:
        # get the uploaded image from the request body
        image = request.files['image']
        print(image.filename)
        #resize image
        # create a new file name for the uploaded image
        filename = secure_filename(image.filename)

        # save the image to the server
        image.save(os.path.join('uploads', filename))

        # apply thresholding and highlight the tumor
        highlighted_image = highlight_tumor(os.path.join('uploads', filename))

        # save the highlighted image
        highlighted_filename = 'highlighted_' + filename
        cv2.imwrite(os.path.join('uploads', highlighted_filename), highlighted_image)
        # evaluate the image using the CNN model
        object.fileDialog(os.path.join('uploads', highlighted_filename))
        # print(object.output)
        os.remove(os.path.join('uploads', filename))
        os.remove(os.path.join('uploads', highlighted_filename))
        # os.remove(os.path.join('uploads', filename)))
            # Convert the processed image to JPEG format
        ret, jpeg = cv2.imencode('.jpg', highlighted_image)
        encoded_image = jpeg.tobytes()

        # Create a response JSON object
        response = {
            'output': object.output,
            'image': base64.b64encode(encoded_image).decode('utf-8')
        }
        
        return jsonify(response)

    except Exception as e:
        # logging.error(f"Error occurred during image upload: {str(e)}")
        print(f"Error occurred during image upload: {str(e)}")
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    
    if(app.run(host='0.0.0.0',port=8080)):
        print("server uis running")
