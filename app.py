from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def counter_thread():
    count = 0
    while True:
        socketio.emit('counter_update', {'count': count})
        count += 1
        time.sleep(1)  # Update every second

# Start the background thread
thread = threading.Thread(target=counter_thread)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5050)
