
from flask import Flask, render_template, Response, redirect, url_for, request
from camera import VideoCamera
from flask_fontawesome import FontAwesome

app = Flask(__name__, static_folder='static')
fa = FontAwesome(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html');

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/upload_video')
def upload_video():
    return render_template('upload_video.html');

@app.route('/video_feed')
def video_feed():
    selectValue = request.form.get('select1')
    print("=================")
    print(selectValue)
    print("=================")
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/about_us')
def about_us():
    return render_template('about_us.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
