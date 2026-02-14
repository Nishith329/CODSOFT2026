from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""

    if request.method == "POST":
        plot = request.form["plot"]
        vec = vectorizer.transform([plot])
        prediction = model.predict(vec)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
