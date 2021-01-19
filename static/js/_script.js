//$( document ).ready(function() {
//  $("#prob_1:contains('Normal')").removeAttr('onClick');
//  $("#prob_1:contains('Normal')").removeAttr('href');
//});

// show file name in file selector
$('input[type="file"]').change(function(e){
  var fileName = e.target.files[0].name;
  $('.custom-file-label').html(fileName);
});

// clear canvas
function clearcanvas1()
{
  var canvas = document.getElementById('chart_1'),
  ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function clearcanvas2()
{
  var canvas = document.getElementById('chart_2'),
  ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// initialize default variable
var prob_1 = 0;
var prob_2 = 0;

$("#prob_1").text('Normal Lungs' + ' ~ ' + prob_1 + '%');
$("#prob_2").text('Pleural Effusion' + ' ~ ' + prob_2 + '%');

// display image
let base64Image;
$("#image-selector").change(function() {
  let reader = new FileReader();
  reader.onload = function(e) {
    let dataURL = reader.result;
    $('#selected-image').attr("src", dataURL);
    base64Image = dataURL.replace("data:image/png;base64,","");
    console.log(base64Image);
  }

  // do this when new image load
  reader.readAsDataURL($("#image-selector")[0].files[0]);
  $("#prob_1").text('Normal Lungs' + ' ~ '  + prob_1 + '%');
  $("#prob_2").text('Pleural Effusion' + ' ~ ' + prob_2 + '%');
  clearcanvas1();
  clearcanvas2();
});

$("#predict-button").click(function(event){
  let message = {
    image: base64Image
  }
  console.log(message);
  $.post("http://127.0.0.1:5000/predict", JSON.stringify(message), function(response){
    $("#prob_1").text(response.prediction.prob_key_1 + ' ~ ' + response.prediction.prob_value_1 + '%');
    $("#prob_2").text(response.prediction.prob_key_2 + ' ~ ' + response.prediction.prob_value_2 + '%');
    console.log(response);

    let predictions = Object.entries(response.prediction).map(function(entry){
      return {
        category : entry[0],
        value : entry[1]
      };
    });
    var data = response.prediction;
    prob_value_1 = data.prob_value_1;
    prob_value_2 = data.prob_value_2;

    var defanofinding = 0;
    var defaeffusion = 0;
    var updatednofinding = prob_value_1;
    var updatedeffusion = prob_value_2;

    var valuenofinding = defanofinding;
    var datanofinding = {
      labels: ["Normal Lungs"],
      datasets: [
        {
          data: [valuenofinding, 100-valuenofinding],
          backgroundColor: [
            "#8a0303",
            "#0b0a13"
          ]
        }]
      };

      var mychartnofinding = new Chart(document.getElementById('chart_1'), {
        type: 'pie',
        data: datanofinding,
        options: {
          responsive: true,
          legend: {
            display: false
          },
          elements: {
            arc: {
              borderWidth: 0
            }
          },
          cutoutPercentage: 40,
          tooltips: {enabled: false},
          hover: {mode: null}

        }
      });

      var valueeffusion = defaeffusion;
      var dataeffusion = {
        labels: ["Pleural Effusion"],
        datasets: [
          {
            data: [valueeffusion, 100-valueeffusion],
            backgroundColor: [
              "#8a0303",
              "#0b0a13"
            ]
          }]
        };

        var mycharteffusion = new Chart(document.getElementById('chart_2'), {
          type: 'pie',
          data: dataeffusion,
          options: {
            responsive: true,
            legend: {
              display: false
            },
            elements: {
              arc: {
                borderWidth: 0
              }
            },
            cutoutPercentage: 40,
            tooltips: {enabled: false},
            hover: {mode: null}

          }
        });

        mychartnofinding.data.datasets[0].data[0] = updatednofinding;
        mychartnofinding.data.datasets[0].data[1] = 100-updatednofinding;
        mycharteffusion.data.datasets[0].data[0] = updatedeffusion;
        mycharteffusion.data.datasets[0].data[1] = 100-updatedeffusion;
        mychartnofinding.update();
        mycharteffusion.update();

      });
    });
