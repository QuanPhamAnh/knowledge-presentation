from src import app
from flask import render_template, redirect, template_rendered, url_for, flash, request
# from market.models import Item, User
# from market.forms import RegisterForm, LoginForm, PurchaseItemForm, SellItemForm
# from market import db
# from flask_login import login_user, logout_user, login_required, current_user

store_questions = ""

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home_page():
    global store_questions
    if request.method == 'POST':
        store_questions = request.form['question']
        return redirect(url_for('search_page'))
    else:
        return render_template('home.html')


@app.route('/search', methods=['GET', 'POST'])
def search_page():
    global store_questions
    if request.method == 'POST':
        question = request.form['question']
    else: 
        question = store_questions
    return render_template('search.html', question = question)


@app.route('/about', methods=['GET', 'POST'])
def about_page():
    return render_template('about.html')


@app.route('/ref', methods=['GET', 'POST'])
def ref_page():
    return render_template('ref.html')







