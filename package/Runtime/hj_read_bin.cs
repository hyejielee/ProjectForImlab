
using System;
using System.IO;
using UnityEngine;

public static class hj_read_bin
{
    const uint F_XYZ = 1 << 0;
    const uint F_ROT = 1 << 1;
    const uint F_SC  = 1 << 2;
    const uint F_OP  = 1 << 3;

    public static void Read(
        string path,
        out int[] indices,
        out Vector3[] xyzDelta,
        out Quaternion[] rot,
        out Vector3[] scale,
        out float[] opacity)
    {
        indices = null;
        xyzDelta = null;
        rot = null;
        scale = null;
        opacity = null;

        if (!File.Exists(path))
        {
            Debug.LogWarning($"BIN not found: {path}");
            return;
        }

        using (var br = new BinaryReader(File.OpenRead(path)))
        {
            //HEADER
            uint magic = br.ReadUInt32();
            uint version = br.ReadUInt32();
            uint frameIdx = br.ReadUInt32();

            int K = (int)br.ReadUInt32();
            int N = (int)br.ReadUInt32();

            uint flags = br.ReadUInt32();

            int rotDim = (int)br.ReadUInt32();
            int scDim = (int)br.ReadUInt32();
            int opDim = (int)br.ReadUInt32();

            Debug.Log($"[BIN] frame={frameIdx} magic=0x{magic:X8} version={version} K={K} N={N} flags={flags} rotDim={rotDim} scDim={scDim} opDim={opDim}");

            //position
            indices = new int[K];
            for (int i = 0; i < K; i++)
                indices[i] = br.ReadInt32();

            xyzDelta = new Vector3[K];
            for (int i = 0; i < K; i++)
            {
                float x = br.ReadSingle();
                float y = br.ReadSingle();
                float z = br.ReadSingle();
                xyzDelta[i] = new Vector3(x, y, z);
            }

            //rotation
            if ((flags & F_ROT) != 0)
            {
                rot = new Quaternion[N];

                for (int i = 0; i < N; i++)
                {
                    float w = ReadHalf(br);
                    float x = ReadHalf(br);
                    float y = ReadHalf(br);
                    float z = ReadHalf(br);

                    //QUEEN 의 원래 순서가 w,x,y,z
                    //근데 유니티에서는 x,y,z,w
                    rot[i] = new Quaternion(x, y, z, w);
                }
            }

            //scale
            if ((flags & F_SC) != 0)
            {
                scale = new Vector3[N];

                for (int i = 0; i < N; i++)
                {
                    float x = ReadHalf(br);
                    float y = ReadHalf(br);
                    float z = ReadHalf(br);

                    scale[i] = new Vector3(x, y, z);
                }
            }

            //opacity
            if ((flags & F_OP) != 0)
            {
                opacity = new float[N];

                for (int i = 0; i < N; i++)
                {
                    opacity[i] = ReadHalf(br);
                }
            }
        }
    }

    // ---------- float16 reader ----------
    static float ReadHalf(BinaryReader br)
    {
        ushort val = br.ReadUInt16();
        return Mathf.HalfToFloat(val);
    }
}

