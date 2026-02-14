from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():
    result = ""

    if request.method == "POST":
        msg = request.form["message"]
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]

        if pred.lower() == "spam":
            result = "⚠ Spam Message"
        else:
            result = "✅ Legitimate Message"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
