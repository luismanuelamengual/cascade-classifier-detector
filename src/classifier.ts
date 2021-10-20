
export class Classifier {

    private tdepth;
    private ntrees;
    private tcodes;
    private tpreds;
    private thresh;

    constructor(cascadeClassifierBase64: string);
    constructor(cascadeClassifierBytes: Int8Array);
    constructor(cascadeClassifierBytesOrBase64: string | Int8Array) {
        let bytes: Int8Array;
        if (cascadeClassifierBytesOrBase64 instanceof Int8Array) {
            bytes = cascadeClassifierBytesOrBase64;
        } else {
            const binaryString = window.atob(cascadeClassifierBytesOrBase64);
            const len = binaryString.length;
            bytes = new Int8Array(len);
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
        }
        const dview = new DataView(new ArrayBuffer(4));
        let p = 8;
        dview.setUint8(0, bytes[p]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
        this.tdepth = dview.getInt32(0, true);
        p = p + 4;
        dview.setUint8(0, bytes[p]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
        this.ntrees = dview.getInt32(0, true);
        p = p + 4;
        const tcodes_ls = [];
        const tpreds_ls = [];
        const thresh_ls = [];
        for (let t = 0; t < this.ntrees; ++t) {
            Array.prototype.push.apply(tcodes_ls, [0, 0, 0, 0]);
            Array.prototype.push.apply(tcodes_ls, bytes.slice(p, p + 4 * Math.pow(2, this.tdepth) - 4));
            p = p + 4 * Math.pow(2, this.tdepth) - 4;
            for (let i = 0; i < Math.pow(2, this.tdepth); ++i) {
                dview.setUint8(0, bytes[p]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
                tpreds_ls.push(dview.getFloat32(0, true));
                p = p + 4;
            }
            dview.setUint8(0, bytes[p]), dview.setUint8(1, bytes[p + 1]), dview.setUint8(2, bytes[p + 2]), dview.setUint8(3, bytes[p + 3]);
            thresh_ls.push(dview.getFloat32(0, true));
            p = p + 4;
        }
        this.tcodes = new Int8Array(tcodes_ls);
        this.tpreds = new Float32Array(tpreds_ls);
        this.thresh = new Float32Array(thresh_ls);
    }

    public process(r, c, s, pixels, ldim) {
        r = 256 * r;
        c = 256 * c;
        let root = 0;
        let o = 0.0;
        const pow2tdepth = Math.pow(2, this.tdepth) >> 0;
        for (let i = 0; i < this.ntrees; ++i) {
            let idx = 1;
            for (let j = 0; j < this.tdepth; ++j) {
                const i1 = ((r + this.tcodes[(root + 4 * idx)] * s) >> 8) * ldim + ((c + this.tcodes[root + 4 * idx + 1] * s) >> 8);
                const i2 = ((r + this.tcodes[root + 4 * idx + 2] * s) >> 8) * ldim + ((c + this.tcodes[root + 4 * idx + 3] * s) >> 8);
                idx = 2 * idx + (pixels[i1] <= pixels[i2] ? 1 : 0);
            }
            o = o + this.tpreds[pow2tdepth * i + idx - pow2tdepth];
            if (o <= this.thresh[i]) {
                return -1;
            }
            root += 4 * pow2tdepth;
        }
        return o - this.thresh[this.ntrees - 1];
    }
}
