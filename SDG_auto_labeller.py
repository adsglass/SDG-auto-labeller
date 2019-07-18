import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from scipy import stats
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def clean_merge_columns(df, columns=["title_name", "findings_outcome", "findings_outcomes_descrip"]):

    """
    convert all columns from systematic reviews into a single columns for easy parsing and clean up
    any special characters
    """

    df = df[columns]
    df_ = df.iloc[:,0]
    for col in list(df.columns)[1:]:
        df_ = df_ + " " + df[col] 
    
    df_ = df_.map(lambda x: x.replace('æ', ' ').replace('ñ', ' ').replace('â', "'").replace("  ", " "))
    
    return pd.DataFrame(df_)

def tokenize_vectorize(ssa, sdg, stopwords=[]):
    
    """
    Tokenizes and TF-IDF vectorizes the list of SDGs. Uses the tokens found in the 
    SDGs to vectorize the systematic reviews. Returns separate dataframes for 
    systematic reviews and SDGs. Uses stop words from NLTK corpus with the option
    to include customer stop words in the "stopwords" keyword argument
    """
    # download stopwords from nltk corpus
    stopwords_ = nltk.corpus.stopwords.words('english')
    stopwords = stopwords + stopwords_
    #
    cvec = TfidfVectorizer(ngram_range=(1, 3), stop_words=stopwords)
    sdg_vec = cvec.fit_transform(sdg)
    ssa_vec = cvec.transform(list(ssa[0]))
    
    df_sdg_vec = pd.DataFrame(sdg_vec.toarray(),
                  columns=cvec.get_feature_names())
    
    df_ssa_vec = pd.DataFrame(ssa_vec.toarray(),
                  columns=cvec.get_feature_names())
    
    print("SDGs and systematic reviews vectorized \n")
    
    return df_ssa_vec, df_sdg_vec

def sort_df(s, num):
    
    """
    Helper function: finds the top-ranked keywords / n-grams and their corresponding 
    TF-IDF scores for each systematic review paper. Returns a Pandas series with tuples: 
    (TF-IDF score, keyword / n-gram). 
    """
    
    tmp = s.sort_values(ascending=False)[:num]  # earlier s.order(..)
    cols = list(tmp.index)
    tmp.index = range(num)
    tmp = pd.Series(list(zip(tmp, cols)))
    return tmp

def select_col(s, col):
    
    """
    Helper function: Selects either the first or second element of a tuple in a Pandas series.
    Returns a Pandas series with just that element as a str or int.
    """
    
    tmp = s.map(lambda x: x[col])
    return tmp

def compare(s, df):
    
    """
    Helper function: Finds the SDG where the keyword / n-gram has the highest TF-IDF score.
    Returns a Pandas series with the SDG.
    """
    
    tmp = select_col(s, 1)
    tmp = tmp.map(lambda x: df[x].idxmax())
    return tmp

def rename_cols(prefix, df):
    
    """
    Helper function: Inserts a specified prefix into the column names of a 
    dataframe. Return the dataframe with updated col names.
    """
    
    df.columns = [prefix + "_" + str(col) for col in df.columns]
    return df

def create_sim_matrix(df_ssa_vec, df_sdg_vec):
    
    """
    Created a cosine similarity matrix between the systematic review papers
    and the SDG goals, based on their TF-IDF vectors. Returns the similarity
    matrix as a Pandas dataframes.
    """
    
    sim_matrix = pd.DataFrame(cosine_similarity(df_ssa_vec, df_sdg_vec))
    return sim_matrix

def extract_feature_df(df_ssa_vec, df_sdg_vec, num_keywords=3):
    
    """
    Extracts features from the systematic review and SDG 
    vectors, including:
    - Label of the SDGs where the top-ranked n (default 3) keywords / n-grams appear most 
    frequently
    - Cosine similarity scores between each review and each SDG
    """
    
    
    keyword_pct_df = df_ssa_vec.T.apply(lambda x: sort_df(x, num_keywords))
    comparison_df = keyword_pct_df.apply(lambda x: compare(x, df_sdg_vec)).T
    sim_df = create_sim_matrix(df_ssa_vec, df_sdg_vec)
    pct_df = keyword_pct_df.apply(lambda x: select_col(x, 0)).T
    
    # rename columns
    
    comparison_df = rename_cols("match_label", comparison_df)
    sim_df = rename_cols("cosine", sim_df)
    pct_df = rename_cols("tfidf", pct_df)
    
    # concat into feature matrix and get dummy variables
    
    X = pd.concat([comparison_df, sim_df, pct_df], axis=1)
    X = pd.get_dummies(X, columns=comparison_df.columns, drop_first=True)
    
    print("Features extracted \n")
    
    return X

def predict_from_model(model_file, X, class_encoder, threshold=0.80):
    
    """
    Loads picked model and uses this to create new labels on an unlabelled
    CSV file. 
    """
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    preds = pd.DataFrame(model.predict(X), columns=["Predicted"])
    pred_proba = model.predict_proba(X)
    
    proba_df = pd.DataFrame(pred_proba)
    proba_df.columns = class_encoder.classes_
    
    final_df = pd.merge(proba_df, preds, left_index=True, right_index=True)

    final_df["max_proba"] = final_df.loc[:, "Aid":"WASH"].max(axis=1)
    final_df["final_label"] = final_df.apply(lambda x: x["Predicted"] if x["max_proba"] > threshold else "NA", axis=1)
    
    final_df["Predicted"] = class_encoder.inverse_transform(final_df["Predicted"].astype('int'))
    
    mask = final_df["final_label"] != "NA"
    final_df.loc[mask, "final_label"] = class_encoder.inverse_transform(final_df[mask]["final_label"].astype('int')).copy()
    
    final_df.to_csv("SDG_labels.csv")
    print("Model saved as 'SDG_labels.csv'")
    
    mask = final_df["final_label"] != "NA"
    print("Number of papers labelled with 95% accuracy: ", len(final_df[mask]))
    

def build_tune_model(Xtrain, Xtest, ytrain, ytest):
    
    """
    Builds an XGBoost model and tunes the hyperparameters using Bayesian hyperopt library.
    Return the best parameters found in 50 trials.
    """
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    import warnings
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
    warnings.filterwarnings("ignore")
    
    
    
    xgb = XGBClassifier(objective='multi:softprob', num_class=len(le.classes_))

    def objective(space):

        xgb = XGBClassifier(n_estimators = 300,
                                max_depth = int(space['max_depth']),
                                min_child_weight = space['min_child_weight'],
                                subsample = space['subsample'],
                                colsample_bytree = space['colsample_bytree'],
                                gamma = space['gamma'],
                            learning_rate = space['learning_rate'], num_class=16
                           )

        scores = cross_val_score(xgb, Xtrain, ytrain, cv=kf)
        ave_score = scores.mean()
        print("ave score: ", ave_score)
        return {'loss':1-ave_score, 'status': STATUS_OK}


    # define hyperparameter search space
    space ={
            'max_depth': hp.quniform("max_depth", 3, 10, 1),
            'min_child_weight': hp.quniform ('min_child', 1, 10, 1),
            'subsample': hp.uniform ('subsample', 0.3, 1),
            'gamma': hp.uniform('gamma', 0.0, 5),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.07)
        }

    # run function and save results / best hyperparmeters
    trials_xgb = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials_xgb)
    
    best_params_xgb = space_eval(space,best)
    
    return best_params_xgb

def build_final_model(Xtrain, ytrain, Xtest, ytest):
    
    """
    Builds the final XGBoost model using best hyperparameters found in build_tune_model
    function. Returns predictions and predicted probabilities.
    """
    
    best_params_xgb = build_tune_model(Xtrain, Xtest, ytrain, ytest)
    
    xgb = XGBClassifier(
        n_estimators = 300,
        max_depth = int(best_params_xgb["max_depth"]), 
        min_child_weight = best_params_xgb["min_child_weight"],
        subsample = best_params_xgb["subsample"], 
        gamma = best_params_xgb["gamma"],
        colsample_bytree = best_params_xgb["colsample_bytree"],
        learning_rate = best_params_xgb['learning_rate'])
    
    xgb.fit(Xtrain, ytrain)
    score = xgb.score(Xtest, ytest)
    preds = xgb.predict(Xtest)
    pred_proba = xgb.predict_proba(Xtest)
    
    print("accuracy score: ", score)
    
    with open('xgb_sdg_model.pickle', 'wb') as f:
        pickle.dump(xgb, f)
        
    return preds, pred_proba

def evaluate_model(preds, pred_proba, ytest):
    
    """
    Evaluates final model, saving a confusion matrix heatmap plot (PNG
    file) and a classification report (as CSV file).
    """
    
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report, \
    precision_recall_fscore_support
    
    def save_confusion_matrix(ytest, preds):
        
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set(font_scale=1)
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(
            cmtrx,
            annot=True,
            fmt='d',    ax=ax)
        plt.xticks(range(0,16), le.classes_, rotation=45)
        plt.yticks(range(0,16), le.classes_, rotation=45)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {model} \n')

        plt.show()

        cmtrx = confusion_matrix(ytest, preds)
        
        fig.savefig('confusion_matrix.png')
        
    def save_classification_report(ytest, preds):
        clf_rep = precision_recall_fscore_support(ytest, preds)
        out_dict = {
                 "precision" :clf_rep[0].round(2)
                ,"recall" : clf_rep[1].round(2)
                ,"f1-score" : clf_rep[2].round(2)
                ,"support" : clf_rep[3]
                }
        out_df = pd.DataFrame(out_dict, index=sdg_headings)
        avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(
            x.sum(), 2)).to_frame().T)
        avg_tot.index = ["avg/total"]
        out_df = out_df.append(avg_tot)
        
        out_df.to_csv("classification_report.csv")
    
    save_confusion_matrix(ytest, preds)
    save_classification_report(ytest, preds)

if __name__ == "__main__":
    
    x = input("Enter path of systematic review papers (leave blank if default): ")

    if x == "":
        ssa = pd.read_stata("SSAevidenceNov_181019.dta", encoding='latin-1')
    else:
        ssa = pd.read_stata(x, encoding='latin-1')

    with open('SDG_list.pickle', 'rb') as f:
        sdg_list = pickle.load(f)

    with open('SDG_headings.pickle', 'rb') as f:
        sdg_headings = pickle.load(f)
        
    with open('SDG_label_encoder.pickle', 'rb') as f:
        le = pickle.load(f)
    
    ssa = clean_merge_columns(ssa)
    df_ssa_vec, df_sdg_vec = tokenize_vectorize(ssa, sdg_list)
    X = extract_feature_df(df_ssa_vec, df_sdg_vec)
    
    PATH = "xgb_sdg_model.pickle"
    
    if not os.path.exists(PATH):
        
        print("No existing model exists")
        train_model = input("Train model (y/n)? ")
        
        if train_model in ["y", "Y", "yes", "Yes"]:
        
            from xgboost import XGBClassifier

            x = input("Enter path of labelled Excel file (leave blank if default): ")

            if x == "":
                labelled_df = pd.read_excel("SDGresults_py.xlsx")
                col = input("Enter column containing labels (leave blank if default): ")
                y = labelled_df[col]

            else:
                labelled_df = pd.read_excel(x)
                y = labelled_df["Final_answer"]

            y = y.dropna()
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)

            with open('SDG_label_encoder.pickle', 'wb') as f:
                pickle.dump(le, f)
                
            from sklearn.model_selection import train_test_split

            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, shuffle=True)

            preds, pred_proba = build_final_model(best_params_xgb, Xtrain, ytrain, Xtest, ytest)

            evaluate_model(preds, pred_proba, ytest)
        
        else:
            print("Exiting program")
  
    predict_from_model('xgb_sdg_model.pickle', X, le)