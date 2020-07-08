let classifier;

let image;

function preload() {
	classifier = ml5.imageClassifier('MobileNet');
	img = loadImage('./images/bird.jpg');

}

function setup() {
	createCanvas(400,400);
	classifier.classify(img, gotResult);
	image(img, 0, 0);
}

function gotResult(error, results) {
	if (error) {
		console.log(error);
	} else {
		console.log(results);
		createDiv(`Label: ${results[0].label}`);
		createDiv(`Conferenc: ${nf(results[0].confidence, 0, 2)}`)
	}
}