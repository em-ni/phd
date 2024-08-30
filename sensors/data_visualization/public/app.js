const ctx = document.getElementById('combinedHistogram').getContext('2d');
const combinedHistogram = new Chart(ctx, {
    plugins: [ChartDataLabels],
    type: 'bar',
    data: {
        labels: ['Values'],
        datasets: [
            {
                label: 'p1_up',
                data: [0],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            },
            {
                label: 'p2_right',
                data: [0],
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            },
            {
                label: 'p3_down',
                data: [0],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: 'p4_left',
                data: [0],
                backgroundColor: 'rgba(153, 102, 255, 0.5)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                min: 0,
                max: 100,
                ticks: {
                    stepSize: 5,
                    callback: function(value, index, values) {
                        return value + ' psi';
                    }
                }
            }
        },
        plugins: {
            datalabels: {
                align: 'end',
                anchor: 'end',
                formatter: function(value, context) {
                    return value ;
                },
                color: '#444',
                font: {
                    weight: 'bold'
                }
            }
        },
        animation: {
            duration: 100
        }
    }
});


const ws = new WebSocket(`ws://${window.location.hostname}:65432`);
ws.onmessage = function (event) {
    // console.log("Received data:", event.data);  // Log the Blob object
    if (event.data instanceof Blob) {
        // Create a new FileReader to handle the Blob
        const reader = new FileReader();
        
        // This event handler is triggered once reading is finished
        reader.onload = function() {
            // console.log("Converted Blob to text:", reader.result);  // Log the text from Blob
            try {
                const data = JSON.parse(reader.result);  // Parse the text as JSON
                // console.log("Parsed data:", data);  // Log parsed data
                
                // Update chart with the parsed data
                combinedHistogram.data.datasets[0].data = [data.p1];
                combinedHistogram.data.datasets[1].data = [data.p2];
                combinedHistogram.data.datasets[2].data = [data.p3];
                combinedHistogram.data.datasets[3].data = [data.p4];
                combinedHistogram.update();
            } catch (error) {
                console.error("Error parsing JSON:", error);
            }
        };
        
        // Start reading the blob as text
        reader.readAsText(event.data);
    } else {
        // Handle non-Blob data (fallback)
        const data = JSON.parse(event.data);
        combinedHistogram.data.datasets[0].data = [data.p1];
        combinedHistogram.data.datasets[1].data = [data.p2];
        combinedHistogram.data.datasets[2].data = [data.p3];
        combinedHistogram.data.datasets[3].data = [data.p4];
        combinedHistogram.update();
    }
};
