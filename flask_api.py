from flask import Flask
import json

app = Flask(__name__)

COMPANIES = {
    1: {
        'name': 'mindee',
        'employees': 10,
        'address': 'bld raspail'
    },
    2: {
        'name': 'apple',
        'employees': 10000,
        'address': '1 infinite loop'
    }
}


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/companies")
def companies():
    return json.dumps(list(COMPANIES.values()))

@app.route("/companies/<int:id_>")
def company(id_):
    try:
        return json.dumps(COMPANIES[id_])
    except KeyError:
        return json.dumps({'error': 'Company not found', 'code': 'err_not_found'}), 404
