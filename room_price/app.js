console.log('Hi, Tensor Flow');

async function getData() {
	const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');
	const houseData = await houseDataReq.json();
	// console.log(houseData);

	const cleaned = houseData.map( house => ({
		price: house.Price,
		rooms: house.AvgAreaNumberofRooms,
	})).filter(house => (house.price != null && house.rooms != null));
	console.log(cleaned)
	return cleaned;
}



async function run() {
	// Load and plot the original input data we are going to train on
	const data = await getData();
	const values = data.map(d => ({
		x: d.rooms,
		y: d.price,
	}))

	tfvis.render.scatterplot(
		{name: 'No. of rooms vs Price'},
		{values},
		{
			xLabel: 'N.o of Rooms',
			yLabel: 'Price',
			height: 300
		}
	);

	const model = createModel();
	tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

function createModel() {
	const model = tf.sequential();

	model.add(tf.layers.dense({inputShape:[1], units: 1, useBias: true}));

	model.add(tf.layers.dense({units:1, useBias:true}));

	return model;
}

function convertToTensors(data) {
	return tf.tidy(() => {
		tf.util.shuffle(data);
	
		const inputs = data.map(d => d.rooms);
		const labels = data.map(d => d.price);
	
		const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
		const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

		const inputMax = inputTensor.max();
		const inputMin = inputTensor.min();

		const labelMax = labelTensor.max();
		const labelMin = labelTensor.min();

		const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
		const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

		return {
			inputs: normalizedInputs,
			labels: normalizedLabels,
			inputMax,
			inputMin,
			labelMax,
			labelMin,
		}

	})
	
}

document.addEventListener('DOMContentLoaded', run)