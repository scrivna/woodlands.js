var woodlands = require('./woodlands.js');

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



// Train a single tree
var dt = new woodlands.DecisionTree(trainingData, class_name, features);

// Evaluate their accuracy on training & test data
console.log('Single Training Accuracy: '+dt.evaluate(trainingData));	// 1
console.log('Single Test Accuracy: '+dt.evaluate(testingData));			// 1

// predict a value
var prediction = dt.predict({ a: 0, b: 1 });
console.log(prediction); // 1


// Train a forest of trees
var rf = new woodlands.RandomForest(trainingData, class_name, features, {
	numTrees: 100,			// how many trees should we use (results are averaged together)
	percentData: 1,			// what percentage of training data should each tree see (bootstrapping) - For larger datasets I find .15 works well
	percentFeatures: 1		// what percentage of features should each tree see (feature bagging) - For larger datasets I find .7 works well
});


// Evaluate our forest with the test data
console.log('Evaluating Forest...');
console.log(JSON.stringify(rf.evaluate(testingData), null, "\t"));


// and predict a value on potentially unseen data

// return percentage of trees that agree on each class
var prediction = rf.predictProbability({ a: 0, b: 1 }); 
console.log(prediction);	// {"1": 1, "0": 0}

// returns a single class that has the consensus vote
var prediction = rf.predictClass({ a: 0, b: 1 });	
console.log(prediction);	// "1"
