#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Api, Resource
from gensim.models.doc2vec import Doc2Vec
import gensim
import json


app = Flask(__name__) #create the Flask app
CORS(app)

d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

@app.route('/getSimilar')
def getSimilar():
    query = request.args.get('query')
    query = query.lower()
    tokens1 = query.split()
    new_vector = d2v_model.infer_vector(tokens1)
    sims2 = d2v_model.docvecs.most_similar([new_vector])
    return json.dumps(sims2)
if __name__ == '__main__':
    app.run(debug=True)

