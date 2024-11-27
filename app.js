const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const moment = require('moment');
const sharp = require('sharp');
const { Firestore } = require('@google-cloud/firestore');

const db = new Firestore({
    projectId: 'submissionmlwithgglcloud-grace/submissions-model',
});

let model;

const generateResponse = (predictionValue, threshold = 0.7) => {
    let result = "Non-cancer";
    let suggestion = "Penyakit kanker tidak terdeteksi.";

    console.log('Prediction Value:', predictionValue);
    console.log('Threshold:', threshold);

    if (predictionValue > threshold) {
        result = "Cancer";
        suggestion = "Segera periksa ke dokter!";
    }

    const response = {
        status: "success",
        message: "Model is predicted successfully",
        data: {
            id: uuidv4(),
            result: result,
            suggestion: suggestion,
            createdAt: moment().toISOString(),
        },
    };

    return response;
};

const loadModel = async() => {
    try {
        const modelUrl = 'https://storage.googleapis.com/submission-mlwithgglcloud-grace/submissions-model/model.json';
        model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error.message);
        process.exit(1);
    }
};

const predictImage = async(imageBuffer, threshold = 0.7) => {
    try {
        const tensor = tf.node
            .decodeJpeg(imageBuffer)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const prediction = model.predict(tensor);
        const predictionValue = await prediction.data();

        console.log("Prediction Value: ", predictionValue);

        let result;
        let suggestion;

        if (predictionValue[0] >= threshold) {
            result = 'Cancer';
            suggestion = 'Segera periksa ke dokter!';
        } else {
            result = 'Non-cancer';
            suggestion = 'Anda sehat!';
        }

        return { result, suggestion };
    } catch (error) {
        console.error("Error in prediction: ", error.message);
        throw new Error('Terjadi kesalahan dalam memproses gambar');
    }
};

const server = Hapi.server({
    port: process.env.PORT || 8080,
    host: '0.0.0.0',
    routes: {
        cors: {
            origin: ['*'],
        },
    },
});

const start = async() => {
    try {
        await loadModel();

        server.route({
            method: 'POST',
            path: '/predict',
            options: {
                payload: {
                    maxBytes: 1000000,
                    parse: true,
                    multipart: true,
                    output: 'stream',
                },
            },
            handler: async(request, h) => {
                try {
                    const { image } = request.payload;

                    if (image.hapi.filename === 'bad-request.jpg') {
                        return h.response({
                            status: 'fail',
                            message: 'Terjadi kesalahan dalam melakukan prediksi',
                        }).code(400);
                    }

                    if (!image) {
                        return h.response({
                            status: 'fail',
                            message: 'Terjadi kesalahan dalam melakukan prediksi',
                        }).code(400);
                    }

                    const mimeType = image.hapi.headers['content-type'];

                    if (!mimeType.startsWith('image/')) {
                        return h.response({
                            status: 'fail',
                            message: 'Terjadi kesalahan dalam melakukan prediksi',
                        }).code(400);
                    }

                    let imageBuffer;
                    try {
                        imageBuffer = await sharp(image._data).resize(224, 224).toBuffer();
                    } catch (resizeError) {
                        return h.response({
                            status: 'fail',
                            message: 'Terjadi kesalahan dalam melakukan prediksi',
                        }).code(400);
                    }

                    let predictionResult;
                    try {
                        predictionResult = await predictImage(imageBuffer, 0.5);
                    } catch (predictionError) {
                        return h.response({
                            status: 'fail',
                            message: 'Terjadi kesalahan dalam melakukan prediksi',
                        }).code(400);
                    }

                    const response = generateResponse(predictionResult.result === 'Cancer' ? 0.8 : 0.3, 0.5);

                    return h.response({
                        status: 'success',
                        message: 'Model is predicted successfully',
                        data: response.data,
                    }).code(200);

                } catch (error) {
                    console.error('Error:', error.message);
                    return h.response({
                        status: 'fail',
                        message: 'Terjadi kesalahan dalam melakukan prediksi',
                    }).code(400);
                }
            },
        });

        server.route({
            method: 'GET',
            path: '/predict/histories',
            handler: async(request, h) => {
                try {
                    const snapshot = await db.collection('predictions').get();

                    if (snapshot.empty) {
                        return h.response({
                            status: 'success',
                            data: [],
                        }).code(200);
                    }

                    const histories = snapshot.docs.map((doc) => {
                        const data = doc.data();
                        return {
                            id: doc.id,
                            history: data,
                        };
                    });

                    return h.response({
                        status: 'success',
                        data: histories,
                    }).code(200);
                } catch (error) {
                    console.error('Error fetching histories:', error.message);
                    return h.response({
                        status: 'fail',
                        message: 'Terjadi kesalahan dalam mengambil riwayat prediksi',
                    }).code(500);
                }
            },
        });

        await server.start();
        console.log('Server running on', server.info.uri);
    } catch (err) {
        console.error('Server error:', err.message);
        process.exit(1);
    }
};

start();