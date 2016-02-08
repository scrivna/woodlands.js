var assert = require('assert');

var woodlands = require('../woodlands.js');

// define our TRAINING data
var trainingData = [
	{ 'a': 0, 'b': 0, 'output': 0 },
	{ 'a': 0, 'b': 1, 'output': 1 },
	{ 'a': 1, 'b': 0, 'output': 1 },
	{ 'a': 1, 'b': 1, 'output': 0 }
];

// define our TEST data
var testingData = [
	{ 'a': 0, 'b': 0, 'output': 0 },
	{ 'a': 0, 'b': 1, 'output': 1 }
];

// define the feature class we want to predict
var class_name = 'output';

// what features of the training data should we use to train on?
var features = ['a','b'];



describe ('Woodlands', function(){
	it('should have RandomForest method', function () {
      assert(woodlands.RandomForest);
    });
    it('should have DecisionTree method', function () {
      assert(woodlands.DecisionTree);
    });
})

describe ('DecisionTree', function(){
	var dt;
	
	before(function() { // runs before all tests in this block
		dt = new woodlands.DecisionTree(trainingData, class_name, features);
	});
	
	it('should train a single tree', function () {
		assert(dt.model);
    });
    it('should evaluate to 1', function () {
		assert.equal(1, dt.evaluate(trainingData));
    });
    it('should predict 1st testing example correctly', function () {
		assert.equal(testingData[0].output, dt.predict(testingData[0]));
    });
    it('should predict 2nd testing example correctly', function () {
		assert.equal(testingData[1].output, dt.predict(testingData[1]));
    });
    it('should provide object of feature importance', function(){
	   assert(typeof dt.featureImportance() == 'object');
    });
    it('should provide JSON model', function(){
	   assert(typeof dt.toJSON() == 'object');
    });
})

describe ('RandomForest', function(){
	var rf;
	
	before(function() { // runs before all tests in this block
		
		// Train a forest of trees
		rf = new woodlands.RandomForest(trainingData, class_name, features, {
			numTrees: 100,			// how many trees should we use (results are averaged together)
			percentData: 1,			// what percentage of training data should each tree see (bootstrapping) - For larger datasets I find .15 works well
			percentFeatures: 1		// what percentage of features should each tree see (feature bagging) - For larger datasets I find .7 works well
		});
	});
	
	it('should have trained 100 trees', function () {
		assert.equal(100, rf.trees.length);
    });
    it('should provide an evaluation report', function () {
		assert(typeof rf.evaluate(trainingData) == 'object');
    });
    it('should have a high fscore ', function () {
		assert(rf.evaluate(trainingData).fscore > .99);
    });
    it('should predict 1st testing example correctly', function () {
		assert.equal(testingData[0].output, rf.predictClass(testingData[0]));
    });
    it('should predict 2nd testing example correctly', function () {
		assert.equal(testingData[1].output, rf.predictClass(testingData[1]));
    });
    it('should return a probability object', function () {
		assert(typeof rf.predictProbability(testingData[1]) == 'object');
    });
    it('should return a high probability for the correct prediction', function () {
		assert(rf.predictProbability(testingData[1])[testingData[1].output] > 0.99);
    });
})