import {Detection} from "./detection";
import {Classifier} from "./classifier";

export class Detector {

    private classifier: Classifier;
    private memoryIndex = 0;
    private memoryBuffer: Array<any> = [];
    private shiftFactor = 0.1;
    private minSize = 100;
    private maxSize = 1000;
    private scaleFactor = 1.1;
    private iouThreshold = 0.2;

    constructor(classifier: Classifier, memoryBufferSize = 1) {
        this.classifier = classifier;
        if (memoryBufferSize > 1) {
            for (let i = 0; i < memoryBufferSize; ++i) {
                this.memoryBuffer.push([]);
            }
        }
    }

    public detect(image: ImageData): Array<Detection> {
        let detections = [];

        // Utilizar el clasificador para detectar objetos en la imagen
        const imagePixels = this.getImagePixels(image);
        let scale = this.minSize;
        while (scale <= this.maxSize) {
            const step = Math.max(this.shiftFactor * scale, 1) >> 0;
            const offset = (scale / 2 + 1) >> 0;
            for (let row = offset; row <= image.height - offset; row += step) {
                for (let column = offset; column <= image.width - offset; column += step) {
                    const score = this.classifier.process(row, column, scale, imagePixels, image.width);
                    if (score > 0.0) {
                        detections.push([row, column, scale, score]);
                    }
                }
            }
            scale = scale * this.scaleFactor;
        }

        // Utilizar el buffer de memoria para obtener las ultimas n detecciones
        if (this.memoryBuffer.length) {
            this.memoryBuffer[this.memoryIndex] = detections;
            this.memoryIndex = (this.memoryIndex + 1) % this.memoryBuffer.length;
            detections = [];
            for (let i = 0; i < this.memoryBuffer.length; ++i) {
                detections = detections.concat(this.memoryBuffer[i]);
            }
        }

        // Utlizar clustering para descartar detecciones del mismo objeto
        detections = detections.sort((a, b) => b[3] - a[3]);
        const assignments = new Array(detections.length).fill(0);
        const clusters = [];
        for (let i = 0; i < detections.length; ++i) {
            if (assignments[i] == 0) {
                let r = 0.0, c = 0.0, s = 0.0, q = 0.0, n = 0;
                for (let j = i; j < detections.length; ++j) {
                    if (this.calculateIOU(detections[i], detections[j]) > this.iouThreshold) {
                        assignments[j] = 1;
                        r = r + detections[j][0];
                        c = c + detections[j][1];
                        s = s + detections[j][2];
                        q = q + detections[j][3];
                        n = n + 1;
                    }
                }
                clusters.push([r / n, c / n, s / n, q]);
            }
        }
        detections = clusters;

        const detectedItems: Array<Detection> = [];
        if (detections && detections.length) {
            detections = detections.filter((detection) => detection[3] > 5).sort((detection1, detection2) => detection1[3] - detection2[3]);
            detections.forEach((detection) => {
                if (detection.length >= 3) {
                    const centerY = detection[0];
                    const centerX = detection[1];
                    const diameter = detection[2];
                    const radius = diameter / 2;
                    detectedItems.push({ center: { x: centerX, y: centerY }, radius });
                }
            });
        }
        return detectedItems;
    }

    private getImagePixels (image: ImageData): Uint8Array {
        const imageData = image.data;
        const imagePixels = new Uint8Array(image.height * image.width);
        for (let r = 0; r < image.height; ++r) {
            for (let c = 0; c < image.width; ++c) {
                imagePixels[r*image.width + c] = (2 * imageData[(r * 4 * image.width + 4 * c)] + 7 * imageData[r * 4 * image.width + 4 * c + 1] + imageData[r * 4 * image.width + 4 * c + 2]) / 10;
            }
        }
        return imagePixels;
    }

    private calculateIOU(det1, det2) {
        const r1 = det1[0], c1 = det1[1], s1 = det1[2];
        const r2 = det2[0], c2 = det2[1], s2 = det2[2];
        const overr = Math.max(0, Math.min(r1 + s1 / 2, r2 + s2 / 2) - Math.max(r1 - s1 / 2, r2 - s2 / 2));
        const overc = Math.max(0, Math.min(c1 + s1 / 2, c2 + s2 / 2) - Math.max(c1 - s1 / 2, c2 - s2 / 2));
        return overr * overc / (s1 * s1 + s2 * s2 - overr * overc);
    }
}
