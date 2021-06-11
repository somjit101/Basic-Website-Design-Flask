from flask import Flask, render_template
import pandas
import prepBankDataset as bankDataPP

app = Flask(__name__)

data_train, data_test, data = bankDataPP.readBankMarketingDataset()
data = bankDataPP.categorize(data)

# --------------------------- Removing Outliers ------------------------------
min_val = data["duration"].min()
max_val = 1500
data = bankDataPP.remove_outliers(df=data, column='duration', minimum=min_val, maximum=max_val)

min_val = data["age"].min()
max_val = 80
data = bankDataPP.remove_outliers(df=data, column='age', minimum=min_val, maximum=max_val)

min_val = data["campaign"].min()
max_val = 6
data = bankDataPP.remove_outliers(df=data, column='campaign', minimum=min_val, maximum=max_val)
# ---------------------------------------------------------------------------------

# ------------------------- Removing Irrelevant Features --------------------------

data = data.drop('default', axis=1)
data = data.drop('poutcome', axis=1)
data = data.drop('contact', axis=1)
data = data.drop(['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)


@app.route('/')
def index():
    return render_template("homepage.html")


@app.route('/<path:path>')
def unified_routing(path):
    if (path.endswith("homepage") or path.endswith("homepage/")):
        return render_template("homepage.html")

    elif (path.endswith("dataset-bank-marketing") or path.endswith("dataset-bank-marketing/")):
        return render_template("dataset-bank-marketing.html")

    elif (path.endswith("dataset-IPL") or path.endswith("dataset-IPL/")):
        return render_template("dataset-IPL.html")

    elif (path.endswith("about") or path.endswith("about/")):
        return render_template("about.html")

    elif (path.endswith("view_bank_parameters") or path.endswith("view_bank_parameters/")):
        return render_template("dataset-bank-marketing-view_bank_parameters.html", data_train=data_train)

    elif (path.endswith("feature_selection") or path.endswith("feature_selection/")):
        return render_template("dataset-bank-marketing-feature_selection.html", data=data)

    elif (path.endswith("view_dataset_head") or path.endswith("view_dataset_head/")):
        num_rows = 20
        return render_template("dataset-bank-marketing-view_dataset_head.html", data_train=data_train, num_rows=num_rows)


# @app.route('/homepage/')
# def homepage():
#     return render_template("homepage.html")
#
# @app.route('/dataset-bank-marketing/')
# def bankMarketing():
#     return render_template("dataset-bank-marketing.html", data_train = data_train, headVals=headVals)
#
# @app.route('/dataset-IPL/')
# def datasetIPL():
#     return render_template("dataset-IPL.html")
#
# @app.route('/about/')
# def about():
#     return render_template("about.html")

if (__name__ == "__main__"):
    app.run(debug=False)
