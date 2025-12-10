from flask import Flask, request, jsonify, render_template
from web_initial_network import run
from web_graphCSV import graphResults
from base64 import b64encode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('web2New.html')

@app.route('/start-experiment', methods=['POST'])
def start_experiment():
    data = request.get_json()
    adversarialPercentage = data.get('adversarial')

    test = data.get('test')
    if test == '0':
        test = 'perceptible-white-right'
    elif test == '25':
        test = 'perceptible-white-random'
    elif test == '50':
        test = 'fdm'
    elif test == '75':
        test = 'random-fdm-val'
    elif test == '100':
        test = 'random-fdm-loc'
    else:
        print('DashboardApp error: invalid testName from front end')

    train = data.get('train')
    if train == '0':
        train = 'perceptible-white-right'
    elif train == '25':
        train = 'perceptible-white-random'
    elif train == '50':
        train = 'fdm'
    elif train == '75':
        train = 'random-fdm-val'
    elif train == '100':
        train = 'random-fdm-loc'
    else:
        print('DashboardApp error: invalid trainName from front end')

    epochs = data.get('epochs')
    experiments = data.get('experiments')


    # log received values
    adversarialPercentage = int(adversarialPercentage) / 100
    print(f"Received adversarialPercentage: {adversarialPercentage}, testName: {test}, trainName: {train} experiment repeats: {epochs}, experiments: {experiments}")
    print('test: running experimenet')
    fileName = f'test{test}_train{train}'
    run(int(experiments), adversarialPercentage, test, train, int(epochs), fileName)
    graph = graphResults(fileName, test)
    graph_base64 = b64encode(graph.getvalue()).decode('utf-8')  # encode in base64
    print('test: DOING Other stuff')


    #  starting an experiment ish
    return jsonify({"status": "Experiment initiated with received parameters", "graph": graph_base64})



if __name__ == '__main__':
    app.run(debug=True)
