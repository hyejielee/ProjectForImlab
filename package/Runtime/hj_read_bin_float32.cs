 using System;
using System.IO;
using UnityEngine;

public static class hj_read_bin_float32
{
    public const uint MAGIC = 0x54505347; // "GSPT"
    public const uint VERSION = 1;

    const uint F_XYZ   = 1u << 0;
    const uint F_ROT   = 1u << 1;
    const uint F_SC    = 1u << 2;
    const uint F_OP    = 1u << 3;
    const uint F_FDC   = 1u << 4;
    const uint F_FREST = 1u << 5;

    const uint REQUIRED_FLAGS = F_XYZ | F_ROT | F_SC | F_OP | F_FDC | F_FREST;

    public struct BinHeader
    {
        public uint magic;
        public uint version;
        public uint frameIdx;

        public int K;
        public int N;
        public uint flags;

        public int rotDim;
        public int scDim;
        public int opDim;
        public int fdcDim;
        public int frestDim;
        public int shDegree;

        public override string ToString()
        {
            return $"frame={frameIdx} magic=0x{magic:X8} version={version} " +
                   $"K={K} N={N} flags={flags} " +
                   $"rotDim={rotDim} scDim={scDim} opDim={opDim} " +
                   $"fdcDim={fdcDim} frestDim={frestDim} shDegree={shDegree}";
        }
    }

    public sealed class FrameData
    {
        public BinHeader header;

        public int[] indices;          // [K]
        public Vector3[] xyzDelta;     // [K]

        public Vector3[] fdc;          // [N]
        public float[] frest;          // [N * frestDim] flat

        public Vector3[] scale;        // [N]
        public Quaternion[] rot;       // [N]
        public float[] opacity;        // [N]

        public float GetFrest(int gaussianIndex, int coeffIndex)
        {
            if (frest == null)
                throw new InvalidOperationException("frest is null.");

            int dim = header.frestDim;

            if (gaussianIndex < 0 || gaussianIndex >= header.N)
                throw new ArgumentOutOfRangeException(nameof(gaussianIndex));

            if (coeffIndex < 0 || coeffIndex >= dim)
                throw new ArgumentOutOfRangeException(nameof(coeffIndex));

            return frest[gaussianIndex * dim + coeffIndex];
        }
    }

    public static FrameData Read(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"BIN not found: {path}", path);

        using (var br = new BinaryReader(File.OpenRead(path)))
        {
            BinHeader h = ReadHeader(br);
            ValidateHeader(h);

            FrameData data = new FrameData();
            data.header = h;

            Debug.Log($"[BIN] {h}");

            // fixed order:
            // header
            // xyz indices[K] int32
            // xyz delta[K,3] float32
            // f_dc[N,3] float32
            // f_rest[N,frestDim] float32
            // sc[N,3] float32
            // rot[N,4] float32   (file: w,x,y,z)
            // op[N,1] float32

            data.indices = ReadInt32Array(br, h.K);
            data.xyzDelta = ReadVector3Array(br, h.K);

            data.fdc = ReadVector3Array(br, h.N);
            data.frest = ReadFloatArray(br, checked(h.N * h.frestDim));
            data.scale = ReadVector3Array(br, h.N);
            data.rot = ReadQuaternionArrayWXYZ(br, h.N);
            data.opacity = ReadFloatArray(br, h.N);

            long remain = br.BaseStream.Length - br.BaseStream.Position;
            if (remain != 0)
                throw new InvalidDataException($"Unread trailing bytes remain: {remain}");

            return data;
        }
    }

    static BinHeader ReadHeader(BinaryReader br)
    {
        BinHeader h = new BinHeader
        {
            magic    = br.ReadUInt32(),
            version  = br.ReadUInt32(),
            frameIdx = br.ReadUInt32(),

            K        = checked((int)br.ReadUInt32()),
            N        = checked((int)br.ReadUInt32()),
            flags    = br.ReadUInt32(),

            rotDim   = checked((int)br.ReadUInt32()),
            scDim    = checked((int)br.ReadUInt32()),
            opDim    = checked((int)br.ReadUInt32()),
            fdcDim   = checked((int)br.ReadUInt32()),
            frestDim = checked((int)br.ReadUInt32()),
            shDegree = checked((int)br.ReadUInt32())
        };

        return h;
    }

    static void ValidateHeader(BinHeader h)
    {
        if (h.magic != MAGIC)
            throw new InvalidDataException($"Magic mismatch: got 0x{h.magic:X8}, expected 0x{MAGIC:X8}");

        if (h.version != VERSION)
            throw new InvalidDataException($"Version mismatch: got {h.version}, expected {VERSION}");

        if (h.flags != REQUIRED_FLAGS)
            throw new InvalidDataException($"Flags mismatch: got {h.flags}, expected {REQUIRED_FLAGS}");

        if (h.K < 0 || h.N < 0)
            throw new InvalidDataException($"Invalid counts: K={h.K}, N={h.N}");

        if (h.fdcDim != 3)
            throw new InvalidDataException($"f_dc dim must be 3, got {h.fdcDim}");

        if (h.scDim != 3)
            throw new InvalidDataException($"sc dim must be 3, got {h.scDim}");

        if (h.rotDim != 4)
            throw new InvalidDataException($"rot dim must be 4, got {h.rotDim}");

        if (h.opDim != 1)
            throw new InvalidDataException($"op dim must be 1, got {h.opDim}");

        if (h.frestDim <= 0)
            throw new InvalidDataException($"f_rest dim must be > 0, got {h.frestDim}");
    }

    static int[] ReadInt32Array(BinaryReader br, int count)
    {
        int byteCount = checked(count * sizeof(int));
        byte[] bytes = br.ReadBytes(byteCount);

        if (bytes.Length != byteCount)
            throw new EndOfStreamException($"Failed to read int[{count}]");

        int[] arr = new int[count];
        Buffer.BlockCopy(bytes, 0, arr, 0, byteCount);
        return arr;
    }

    static float[] ReadFloatArray(BinaryReader br, int count)
    {
        int byteCount = checked(count * sizeof(float));
        byte[] bytes = br.ReadBytes(byteCount);

        if (bytes.Length != byteCount)
            throw new EndOfStreamException($"Failed to read float[{count}]");

        float[] arr = new float[count];
        Buffer.BlockCopy(bytes, 0, arr, 0, byteCount);
        return arr;
    }

    static Vector3[] ReadVector3Array(BinaryReader br, int count)
    {
        float[] raw = ReadFloatArray(br, checked(count * 3));
        Vector3[] arr = new Vector3[count];

        int p = 0;
        for (int i = 0; i < count; i++)
        {
            arr[i] = new Vector3(raw[p], raw[p + 1], raw[p + 2]);
            p += 3;
        }

        return arr;
    }

    static Quaternion[] ReadQuaternionArrayWXYZ(BinaryReader br, int count)
    {
        float[] raw = ReadFloatArray(br, checked(count * 4));
        Quaternion[] arr = new Quaternion[count];

        int p = 0;
        for (int i = 0; i < count; i++)
        {
            float w = raw[p + 0];
            float x = raw[p + 1];
            float y = raw[p + 2];
            float z = raw[p + 3];
            p += 4;

            arr[i] = new Quaternion(x, y, z, w);
        }

        return arr;
    }
}