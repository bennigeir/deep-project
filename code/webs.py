from flask import Flask, redirect, url_for, render_template, request, jsonify
from main import tweet_analysis


app = Flask(__name__)


@app.route('/api/v1/analyse_tweet', methods=['GET'])
def get_analysis():
    tweet = str(request.args.get('tweet', ''))
    model_type = str(request.args.get('model_type', ''))

    # default CNN, TODO refactor...
    if len(model_type) == 0:
        model_type = 'cnn'

    ans = tweet_analysis(tweet, model_type)
    return jsonify({'category': ans})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True, host='localhost', use_reloader=False)