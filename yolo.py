from flask import Flask, redirect, url_for, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
  return "<h1>Welcome to CodingX</h1>"


if __name__ == '__main__':
	app.run(debug=True)