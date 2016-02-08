/*
woodlands.js

Author: Ross Scrivener
Website: http://rossscrivener.co.uk

Description:
A random forest implementation on top of the ID3 Classifier. Using feature bagging & bootstrapping.
*/


var _ = require('lodash');

/**
 * @module Woodlands
 */
module.exports = {
	RandomForest: RandomForest,
	DecisionTree: ID3
};


function RandomForest(_s, target, features, opts){
	
	this.numTrees = opts.numTrees || 100;
	this.percentData = opts.percentData || .2;
	this.percentFeatures = opts.percentFeatures || .7;
	this.verbose = opts.verbose || false;
	
	this.data = _s;
	this.target = target;
	this.features = features.slice(0);
	this.trees = [];
	
	for (var i=0; i < this.numTrees; i++){
		
		// select n% of data
		var d = _s.slice(0);
		d = _.slice(_.shuffle(d), 0, (d.length * this.percentData));
		var n_features = Math.round(features.length * this.percentFeatures);
		
		var f = features.slice(0);
		f = _.slice(_.shuffle(f), 0, n_features);
		
		if (this.verbose){
			console.log('Tree '+i+' : '+d.length+' data / '+f.length+' features');
			console.log(JSON.stringify(f.sort()));
		}
		
		
		this.trees.push(new ID3(d, target, f));
	}
}

RandomForest.prototype.predictClass = function(sample){
	return this.predict(sample, 'class');
};
RandomForest.prototype.predictProbability = function(sample){
	return this.predict(sample, 'probability');
};

RandomForest.prototype.predict = function(sample, type){
	
	type = type || 'class'; // class or probability
	
	var results = [];
	_.each(this.trees, function(dt){
		results.push(dt.predict(sample));
	});
	
	//console.log(results);
	
	if (type == 'class') return mostCommon(results);
	if (type == 'probability'){
		var counts = {};
		for(var i = 0; i< results.length; i++){
		    var num = results[i];
		    counts[num] = counts[num] ? counts[num]+1 : 1;
		}
		_.each(counts, function(e, i){
			counts[i] = e / results.length;
		});
		return counts;
	}
	
	return results.reduce(function(a, b){return a+b;}) / results.length;
};
RandomForest.prototype.evaluate = function(samples){
    var recall_size = 0;
    var recall_correct = 0;
    var predict_correct = 0;
    var predict_size = 0;
    var instance = this;
    
    var report = {
	    size: 0,
	    correct:0,
	    incorrect:0,
	    accuracy: 0,
	    precision: 0,
	    recall: 0,
	    fscore: 0,
	    class:{},
	    featureImportance: null
    };
    
    _.each(samples, function(s) {
	    
	    report.size++;
	    
		var pred = instance.predictClass(s);
		var actual = s[instance.target];
		
		report.class[pred] = report.class[pred] || {size:0, predicted:0, predicted_correct:0};
		report.class[pred].predicted++;
		
		report.class[actual] = report.class[actual] || {size:0, predicted:0, predicted_correct:0};
		report.class[actual].size++;
		
		
		if(pred == actual) {
			report.correct++;
			report.class[pred].predicted_correct++;
		} else {
			report.incorrect++;
		}
		
    });
    
    var class_length = 0;
    _.each(report.class, function(d) {
	    d.precision = d.predicted_correct / d.predicted;
	    d.recall = d.predicted_correct / d.size;
	    d.fscore = 2 * (d.precision * d.recall) / (d.precision + d.recall);
	    
	    report.precision+=d.precision;
	    report.recall+=d.recall;
	    report.fscore+=d.fscore;
	    
	    class_length++;
	});
    
    report.accuracy = report.correct / report.size;
    report.precision/=class_length;
    report.recall/=class_length;
    report.fscore/=class_length;
    
    report.featureImportance = this.featureImportance();
    return report;
};
RandomForest.prototype.featureImportance = function(){
	var r = {};
	for (var i in this.features){
		r[this.features[i]] = gain(this.data,this.target,this.features[i]);
	}
	return r;
};








/*
ID3 Decision Tree Algorithm

Copyright (c) 2014, Ankit Kuwadekar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* The name Ankit Kuwadekar may not be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANKIT KUWADEKAR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

function ID3(_s, target, features) {
  this.data = _s;
  this.target = target;
  this.features = features;
  
  this.model = createTree(_s, target, features);
}

ID3.prototype = {
  predict: function(sample) {
    var root = this.model;
    while (root.type !== "result") {
      var attr = root.name;
      var sampleVal = sample[attr];
      var childNode = _.find(root.vals,function(x) { return x.name == sampleVal });
      if (childNode){
	      root = childNode.child;
	  } else {
		  try {
		      root = root.vals[0].child;
		  } catch(e){
			  console.log(root);
		  }
      }
    }

    return root.val;
  },

  evaluate: function(samples) {
    var instance = this;
    var target = this.target;

    var total = 0;
    var correct = 0;
    
    _.each(samples, function(s) {
      total++;
      var pred = instance.predict(s);
      var actual = s[target];
      if(pred == actual) {
        correct++;
      }
    });

    return correct / total;
  },

  featureImportance: function(){
	  var r = {};
	  for (var i in this.features){
		  r[this.features[i]] = gain(this.data,this.target,this.features[i]);
	  }
	  return r;
  },
  
  toJSON: function() {
    return this.model;
  }
};


/**
 * Private API
 */

function createTree(_s, target, features) {
  var targets = _.uniq(_.map(_s, target));
  if (targets.length == 1){
    // console.log("end node! "+targets[0]);
    return {type:"result", val: targets[0], name: targets[0],alias:targets[0]+randomTag() }; 
  }
  if(features.length == 0){
    // console.log("returning the most dominate feature!!!");
    var topTarget = mostCommon(targets);
    return {type:"result", val: topTarget, name: topTarget, alias: topTarget+randomTag()};
  }
  var bestFeature = maxGain(_s,target,features);
  var remainingFeatures = _.without(features,bestFeature);
  var possibleValues = _.uniq(_.map(_s, bestFeature));
  
  if (possibleValues.length == 0){
	//console.log("returning the most dominate feature!!!");
    var topTarget = mostCommon(targets);
    return {type:"result", val: topTarget, name: topTarget, alias: topTarget+randomTag()};
  }
  
  var node = {name: bestFeature,alias: bestFeature+randomTag()};
  node.type = "feature";
  node.vals = _.map(possibleValues,function(v){
    // console.log("creating a branch for "+v);
    var _newS = _s.filter(function(x) {return x[bestFeature] == v});
    var child_node = {name:v,alias:v+randomTag(),type: "feature_value"};
    child_node.child =  createTree(_newS,target,remainingFeatures);
    return child_node;
  });
  
  return node;
}

function entropy(vals){
  var uniqueVals = _.uniq(vals);
  var probs = uniqueVals.map(function(x){return prob(x,vals)});
  var logVals = probs.map(function(p){return -p*log2(p) });
  return logVals.reduce(function(a,b){return a+b},0);
}

function gain(_s,target,feature){
  var attrVals = _.uniq(_.map(_s, feature));
  var setEntropy = entropy(_.map(_s, target));
  var setSize = _.size(_s);
  var entropies = attrVals.map(function(n){
    var subset = _s.filter(function(x){return x[feature] === n});
    return (subset.length/setSize)*entropy(_.map(subset,target));
  });
  var sumOfEntropies =  entropies.reduce(function(a,b){return a+b},0);
  return setEntropy - sumOfEntropies;
}

function maxGain(_s,target,features){
  return _.max(features,function(e){return gain(_s,target,e)});
}

function prob(val,vals){
  var instances = _.filter(vals,function(x) {return x === val}).length;
  var total = vals.length;
  return instances/total;
}

function log2(n){
  return Math.log(n)/Math.log(2);
}

function mostCommon(l) {
  return  _.sortBy(l,function(a){
    return count(a,l);
  }).reverse()[0];
}

function count(a, l) {
  return _.filter(l,function(b) { return b === a}).length
}

function randomTag() {
  return "_r"+Math.round(Math.random()*1000000).toString();
}