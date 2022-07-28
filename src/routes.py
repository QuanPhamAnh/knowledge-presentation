from ast import Try
from src import app
from flask import render_template, redirect, url_for, session, request
from flask_session import Session


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        session['store_questions'] = request.form['question']
        return redirect(url_for('search_page'))
    else:
        return render_template('home.html')


@app.route('/search', methods=['GET', 'POST'])
def search_page():
    if request.method == 'POST':
        question = request.form['question']
    else:
        try:
            question = session['store_questions']
        except:
            question = ''
    return render_template('search.html', question = question)


@app.route('/about', methods=['GET', 'POST'])
def about_page():
    return render_template('about.html')


@app.route('/ref', methods=['GET', 'POST'])
def ref_page():
    return render_template('ref.html')







