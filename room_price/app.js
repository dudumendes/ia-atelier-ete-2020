console.log('Hi, Tensor Flow');

async function getData() {
	const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');  
	const houseData = await houseDataReq.json();  
	const cleaned = houseData.map(house => ({
	  price: house.Price,
	  rooms: house.AvgAreaNumberofRooms,
	}))
	.filter(house => (house.price != null && house.rooms != null));
  
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
}

document.addEventListener('DOMContentLoaded', run)