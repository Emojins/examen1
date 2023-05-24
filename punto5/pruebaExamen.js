const model = tf.sequential()

async function Entrenar() {
    const repeticiones = parseInt(document.getElementById('repeticiones').value);
     const epochs = repeticiones;

    model.add(tf.layers.dense({units: 1, inputShape: [1]}));


    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
     //entrenando con formula y= 2x + 8
    const xs = tf.tensor2d([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ], [15, 1]);
    const ys = tf.tensor2d([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], [15, 1]);


    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
             console.log(logs);
             console.log("/n");
             console.log(`Epoch ${epoch+1} - Loss: ${logs.loss.toFixed(4)},`);
          }
        }
      });

      // Imprimir la pérdida final
      console.log(`Final Loss: ${history.history.loss[epochs-1].toFixed(4)}`);

      alert("terminó de entrenar");
}


async function Predecir() {
    const prediccionValor = parseInt(document.getElementById('valorPredecir').value);

    document.getElementById('Resultado').innerText =
    model.predict(tf.tensor2d([prediccionValor], [1, 1])).dataSync();


}

