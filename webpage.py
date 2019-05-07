from flask import Flask, render_template, Response,request
from webcam import WebCam
import os

# TODO: trash collector
# TODO: start, stop, save record
# TODO: Buttons
# TODO: More Features (Login, Welcome, About)
# TODO: log site visitors
# add methods=['GET', 'POST']

app = Flask(__name__)
path = "C:\\Users\\xil14\\PycharmProjects\\webcam"
cascade_path = path + '\\resources\\haarcascade_frontalface_default.xml'
tolerance = 3


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo', methods=['POST'])
def demo():
    return render_template('demo.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/attractive_female')
def attractive_female():
    return render_template('attractive_female.html', title='Attractive Female')


@app.route('/attractive_female_start')
def attractive_female_start():
    return render_template('attractive_female_start.html', title='Attractive Female - now streaming')


@app.route('/attractive_female_func')
def attractive_female_func():
    category = 'Attractive Female'
    learner_path = path + '\\resources\\models\\attractive_female_resnet50.pkl'
    reply = True
    cam = WebCam(category, learner_path, cascade_path, tolerance, reply)
    return Response(cam.feed(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/attractive_male')
def attractive_male_func():
    return render_template('attractive_male.html', title='Attractive Male')


@app.route('/attractive_male_func')
def attractive_male():
    category = 'Attractive Male'
    learner_path = path + '\\resources\\models\\attractive_male_resnet50.pkl'
    cam = WebCam(category, learner_path, cascade_path, tolerance)
    return Response(cam.feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cwd')
def login():
    path = os.getcwd()
    return path


@app.route('/user/<username>')
def profile(username):
    return '{}\'s profile'.format(username)


if __name__ == '__main__':
    app.run(debug=True)  # set false in production
