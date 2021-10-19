
export class DetectionMemoryBuffer {

    private index = 0;
    private memory: Array<any> = [];

    constructor(size: number) {
        for (let i = 0; i < size; ++i) {
            this.memory.push([]);
        }
    }

    public size() {
        return this.memory.length;
    }

    public addDetections(detections: any) {
        this.memory[this.index] = detections;
        this.index = (this.index + 1) % this.memory.length;
    }

    public getDetections(): Array<any> {
        let detections = [];
        for (let i = 0; i < this.memory.length; ++i) {
            detections = detections.concat(this.memory[i]);
        }
        return detections;
    }
}
