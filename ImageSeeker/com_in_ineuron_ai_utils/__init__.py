'''
@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")'''