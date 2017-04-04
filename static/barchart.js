Morris.Bar({
	element: 'bar-chart',
  	data: [
    	{ x: 'V. Pos', y: 100},
    	{ x: 'Pos', y: 75},
    	{ x: 'Neutral', y: 50},
    	{ x: 'Neg', y: 75},
    	{ x: 'V.Neg', y: 50},
  	],
  
  	xkey: 'x',
  	ykeys: ['y'],
  	labels: ['# Tweets'],
	barColors: function (row, series, type) {
	console.log("--> "+row.label, series, type);
	if(row.label == "V. Pos") return "#3300FF";
	else if(row.label == "Pos") return "#6600FF";
	else if(row.label == "Neutral") return "#9900FF";
	else if(row.label == "Neg") return "#FF3333";
	else return "#ff0000";
	}
});