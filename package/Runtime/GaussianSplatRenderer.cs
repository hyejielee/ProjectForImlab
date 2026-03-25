// SPDX-License-Identifier: MIT
// SH 계수 반영 수정사항  - OOO
using System;
using System.IO; 
using GaussianSplatting.Runtime; 
using System.Runtime.InteropServices;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    class GaussianSplatRenderSystem
    {

        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
            {
                if (m_CameraCommandBuffersDone != null)
                {
                    if (m_CommandBuffer != null)
                    {
                        foreach (var cam in m_CameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                        }
                    }
                    m_CameraCommandBuffersDone.Clear();
                }

                m_ActiveSplats.Clear();
                m_CommandBuffer?.Dispose();
                m_CommandBuffer = null;
                Camera.onPreCull -= OnPreCullCamera;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by order and depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var orderA = a.Item1.m_RenderOrder;
                var orderB = b.Item1.m_RenderOrder;
                if (orderA != orderB)
                    return orderB.CompareTo(orderA);
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material SortAndRenderSplats(Camera cam, CommandBuffer cmb)
        {
            Material matComposite = null;
            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                gs.EnsureMaterials();
                matComposite = gs.m_MatComposite;
                var mpb = kvp.Item2;

                // sort
                var matrix = gs.transform.localToWorldMatrix;

                if (gs.m_FrameCounter % gs.m_SortNthFrame == 0)
                    gs.SortPoints(cmb, cam, matrix);
                ++gs.m_FrameCounter;

                // cache view
                kvp.Item2.Clear();
                Material displayMat = gs.m_RenderMode switch
                {
                    GaussianSplatRenderer.RenderMode.DebugPoints => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugPointIndices => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugBoxes => gs.m_MatDebugBoxes,
                    GaussianSplatRenderer.RenderMode.DebugChunkBounds => gs.m_MatDebugBoxes,
                    _ => gs.m_MatSplats
                };
                if (displayMat == null)
                    continue;

                gs.SetAssetDataOnMaterial(mpb);
                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatChunks, gs.m_GpuChunks);

                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);


                //HJ //나라는 버퍼가 있음을 알림.
                mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuOrderBuffer);
                mpb.SetBuffer("_GSBuffer", gs.GpuGSBuffer);
                //mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);
                //
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatScale, gs.m_SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatOpacityScale, gs.m_OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatSize, gs.m_PointDisplaySize);
                //HJ
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOrder, gs.m_SHOrder);
                //
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOnly, gs.m_SHOnly ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayIndex, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayChunks, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds ? 1 : 0);

                cmb.BeginSample(s_ProfCalcView);
                gs.CalcViewData(cmb, cam);
                cmb.EndSample(s_ProfCalcView);

                // draw
                int indexCount = 6;
                int instanceCount = gs.splatCount;
                MeshTopology topology = MeshTopology.Triangles;
                if (gs.m_RenderMode is GaussianSplatRenderer.RenderMode.DebugBoxes or GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    instanceCount = gs.m_GpuChunksValid ? gs.m_GpuChunks.count : 0;

                cmb.BeginSample(s_ProfDraw);
                cmb.DrawProcedural(gs.m_GpuIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(s_ProfDraw);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer {name = "RenderGaussianSplats"};
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            InitialClearCmdBuffer(cam);

            m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
            m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
            m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

            // We only need this to determine whether we're rendering into backbuffer or not. However, detection this
            // way only works in BiRP so only do it here.
            m_CommandBuffer.SetGlobalTexture(GaussianSplatRenderer.Props.CameraTargetTexture, BuiltinRenderTextureType.CameraTarget);

            // add sorting, view calc and drawing commands for each splat object
            Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

            // compose
            m_CommandBuffer.BeginSample(s_ProfCompose);
            m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
            m_CommandBuffer.EndSample(s_ProfCompose);
            m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {

        public enum RenderMode
        {
            Splats,
            DebugPoints,
            DebugPointIndices,
            DebugBoxes,
            DebugChunkBounds,
        }
        public GaussianSplatAsset m_Asset;
        
        //SJ
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 48)]

        public struct SplatCustom
        {
            [FieldOffset(0)] public Vector3 posDelta;    // 12 bytes
            [FieldOffset(12)] public Quaternion rot; // 16 bytes
            [FieldOffset(28)] public Vector3 scale;  // 12 bytes
            [FieldOffset(40)] public float opacity;  // 4 bytes
            [FieldOffset(44)] public float pad;           // 4 bytes
        }    

        [Tooltip("Rendering order compared to other splats. Within same order splats are sorted by distance. Higher order splats render 'on top of' lower order splats.")]
        public int m_RenderOrder;
        //2->20
        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;
        [Tooltip("Show only Spherical Harmonics contribution, using gray color")]
        public bool m_SHOnly;
        [Range(1,30)] [Tooltip("Sort splats only every N frames")]
        public int m_SortNthFrame = 1;

        public RenderMode m_RenderMode = RenderMode.Splats;
        [Range(1.0f,15.0f)] public float m_PointDisplaySize = 3.0f;

        public GaussianCutout[] m_Cutouts;

        public Shader m_ShaderSplats;
        public Shader m_ShaderComposite;
        public Shader m_ShaderDebugPoints;
        public Shader m_ShaderDebugBoxes;
        [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader m_CSSplatUtilities;

        int m_SplatCount; // initially same as asset splat count, but editing can change this
        GraphicsBuffer m_GpuSortDistances;
        internal GraphicsBuffer m_GpuSortKeys;
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuIndexBuffer;

        // these buffers are only for splat editing, and are lazily created
        GraphicsBuffer m_GpuEditCutouts;
        GraphicsBuffer m_GpuEditCountsBounds;
        GraphicsBuffer m_GpuEditSelected;
        GraphicsBuffer m_GpuEditDeleted;
        public GraphicsBuffer GpuEditDeleted => m_GpuEditDeleted;
        GraphicsBuffer m_GpuEditSelectedMouseDown; // selection state at start of operation
        GraphicsBuffer m_GpuEditPosMouseDown; // position state at start of operation
        GraphicsBuffer m_GpuEditOtherMouseDown; // rotation/scale state at start of operation
        

        //HJ
        GraphicsBuffer m_GpuGSBuffer;
        internal GraphicsBuffer GpuGSBuffer => m_GpuGSBuffer; 
        internal GraphicsBuffer m_GpuOrderBuffer;

        GraphicsBuffer m_GpuDeltaIndices;
        GraphicsBuffer m_GpuDeltaXYZ;

        GraphicsBuffer m_GpuFrameRot; //각 버퍼에 미리 담고 .compute에서 합쳐서 담을 겁니다.
        GraphicsBuffer m_GpuFrameScale;
        GraphicsBuffer m_GpuFrameOpacity;
        
        int[] m_TmpIndices;
        Vector3[] m_TmpDeltaXYZ;
        Quaternion[] m_TmpRot;
        Vector3[] m_TmpScale;
        float[] m_TmpOpacity;

        //OOO
        Texture2D m_GpuColorDataTex;
        ushort[] m_TmpColorHalf;
        uint[] m_TmpPackedSH;
        float[] m_TmpColorFloat;
        
        //수정 가능
        const int kFrameFdcDim = 3;
        const int kFrameFRestDim = 24;
        //
        
        int m_LastDeltaK = 0;
        internal int LastDeltaK => m_LastDeltaK;
        int m_DeltaFrame = 1;
        internal int DeltaFrame => m_DeltaFrame;

        public string deltaRoot = @"C:\GS_Project\Queen_pretrain\Packets";

        int m_LastLoadedDeltaFrame = -1;
        int m_LastAppliedDeltaFrame = -1;
        int m_KernelApplyDelta = -1;
        int m_KernelApplyDense = -1;

        
        int KernelApplyDelta()
        {
            if (m_KernelApplyDelta < 0){
                m_KernelApplyDelta = m_CSSplatUtilities.FindKernel("CSApplyIncrementalDeltaSparse");
            }
            return m_KernelApplyDelta;
        }

        
        int KernelApplyDense()
        {
            if (m_KernelApplyDense < 0)
            {
                m_KernelApplyDense = m_CSSplatUtilities.FindKernel("CSApplyDenseAttributes");
            }
            return m_KernelApplyDense;
        }
        

        void ApplyDeltaSparseOnceOnGPU(int frameNumber)
        {
            if (m_LastLoadedDeltaFrame != frameNumber) return;
            if (m_LastAppliedDeltaFrame == frameNumber) return;

            int K = m_LastDeltaK;
            if (K <= 0)
            {
                return;
            }

            int kernel = KernelApplyDelta();

            using var cmb = new CommandBuffer { name = $"ApplyDeltaSparseOnce f{frameNumber}" };

            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_GSBuffer", m_GpuGSBuffer);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_DeltaIndices", m_GpuDeltaIndices);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_DeltaXYZ", m_GpuDeltaXYZ);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_K", K);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_SplatCount", m_SplatCount);

            m_CSSplatUtilities.GetKernelThreadGroupSizes(kernel, out uint tgx, out _, out _);
            int groups = (K + (int)tgx - 1) / (int)tgx;

            cmb.DispatchCompute(m_CSSplatUtilities, kernel, groups, 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

            m_LastDeltaK = 0; 
        }

        
        void ApplyDenseAttributesOnceOnGPU(int frameNumber)
        {
            int kernel = KernelApplyDense();

            using var cmb = new CommandBuffer { name = $"ApplyDenseAttributes f{frameNumber}" };

            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_GSBuffer", m_GpuGSBuffer);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_FrameRot", m_GpuFrameRot);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_FrameScale", m_GpuFrameScale);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernel, "_FrameOpacity", m_GpuFrameOpacity);
            
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_SplatCount", m_SplatCount);

            m_CSSplatUtilities.GetKernelThreadGroupSizes(kernel, out uint tgx, out _, out _);
            int groups = (m_SplatCount + (int)tgx - 1) / (int)tgx;

            cmb.DispatchCompute(m_CSSplatUtilities, kernel, groups, 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

        }
        //


        GpuSorting m_Sorter;
        GpuSorting.Args m_SorterArgs;

        internal Material m_MatSplats;
        internal Material m_MatComposite;
        internal Material m_MatDebugPoints;
        internal Material m_MatDebugBoxes;

        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;
        bool m_Registered;

        static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

        internal static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatSelectedBits = Shader.PropertyToID("_SplatSelectedBits");
            public static readonly int SplatDeletedBits = Shader.PropertyToID("_SplatDeletedBits");
            public static readonly int SplatBitsValid = Shader.PropertyToID("_SplatBitsValid");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int SrcBuffer = Shader.PropertyToID("_SrcBuffer");
            public static readonly int DstBuffer = Shader.PropertyToID("_DstBuffer");
            public static readonly int BufferSize = Shader.PropertyToID("_BufferSize");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int CameraTargetTexture = Shader.PropertyToID("_CameraTargetTexture");
            public static readonly int SelectionCenter = Shader.PropertyToID("_SelectionCenter");
            public static readonly int SelectionDelta = Shader.PropertyToID("_SelectionDelta");
            public static readonly int SelectionDeltaRot = Shader.PropertyToID("_SelectionDeltaRot");
            public static readonly int SplatCutoutsCount = Shader.PropertyToID("_SplatCutoutsCount");
            public static readonly int SplatCutouts = Shader.PropertyToID("_SplatCutouts");
            public static readonly int SelectionMode = Shader.PropertyToID("_SelectionMode");
            public static readonly int SplatPosMouseDown = Shader.PropertyToID("_SplatPosMouseDown");
            public static readonly int SplatOtherMouseDown = Shader.PropertyToID("_SplatOtherMouseDown");

            //OOO
            public static readonly int UseFrameData = Shader.PropertyToID("_UseFrameData");
            //
        }

        [field: NonSerialized] public bool editModified { get; private set; }
        [field: NonSerialized] public uint editSelectedSplats { get; private set; }
        [field: NonSerialized] public uint editDeletedSplats { get; private set; }
        [field: NonSerialized] public uint editCutSplats { get; private set; }
        [field: NonSerialized] public Bounds editSelectedBounds { get; private set; }

        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            SetIndices,
            CalcDistances,
            CalcViewData,
            UpdateEditData,
            InitEditData,
            ClearBuffer,
            InvertSelection,
            SelectAll,
            OrBuffers,
            SelectionUpdate,
            TranslateSelection,
            RotateSelection,
            ScaleSelection,
            ExportData,
            CopySplats,
        }

        //HJ
        readonly Dictionary<KernelIndices, int> m_KernelCache = new Dictionary<KernelIndices, int>();

        int K(KernelIndices k)
        {
            if (m_CSSplatUtilities == null) return -1;

            if (!m_KernelCache.TryGetValue(k, out int id))
            {
                string name = "CS" + k.ToString();
                id = m_CSSplatUtilities.FindKernel(name);
                m_KernelCache[k] = id;
            }
            return id;
        }
        //

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        const int kGpuViewDataSize = 40;

        void CreateResourcesForAsset()
        {   

            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;

            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());

            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());

            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());
            
            //SJ
            m_GpuGSBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 48)
            { name = "GpuGSBuffer" };

            var initAccum = new SplatCustom[m_SplatCount];
            for (int i = 0; i < m_SplatCount; i++)
            {
                initAccum[i].posDelta = Vector3.zero;
                initAccum[i].rot = new Quaternion(0f, 0f, 0f, 1f);
                initAccum[i].scale = Vector3.zero; 
                initAccum[i].opacity = 0f;           
                initAccum[i].pad = 0f;
            }
            m_GpuGSBuffer.SetData(initAccum);
            

            //HJ
            m_GpuDeltaIndices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 4)
            { name = "FrameDeltaIndices" };
            m_GpuDeltaXYZ = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 12)
            { name = "FrameDeltaXYZ" };
            m_GpuFrameRot = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 16)
            { name = "FrameRot" };
            m_GpuFrameScale = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 12)
            { name = "FrameScale" };
            m_GpuFrameOpacity = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 4)
            { name = "FrameOpacity" };
            
            //

            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            
            //OOO
            //tex.Apply(false, true);
            //m_GpuColorData = tex;
            tex.Apply(false, false);

            m_GpuColorData = tex;
            m_GpuColorDataTex = tex;

            m_TmpColorHalf = new ushort[texWidth * texHeight * 4];
            m_TmpPackedSH = new uint[m_GpuSHData.count];
            //

            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize);
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });

            InitSortBuffers(splatCount);
        }
 

        void InitSortBuffers(int count)
        {
            m_GpuSortDistances?.Dispose();
            m_GpuSortKeys?.Dispose();
            m_SorterArgs.resources.Dispose();

            EnsureSorterAndRegister();

            //HJ
            m_GpuSortDistances = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopySource,
                count, 4) { name = "GaussianSplatSortDistances" };

            m_GpuSortKeys = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopySource | GraphicsBuffer.Target.CopyDestination,
                count, 4) { name = "GaussianSplatSortIndices" };

            m_GpuOrderBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopyDestination | GraphicsBuffer.Target.CopySource,
                count, 4) { name = "GaussianSplatOrderBuffer" };

            
            int k = K(KernelIndices.SetIndices);

            m_CSSplatUtilities.SetBuffer(k, Props.OrderBuffer, m_GpuOrderBuffer);
            //

            m_CSSplatUtilities.SetBuffer(k, Props.SplatSortKeys, m_GpuSortKeys); 

            m_CSSplatUtilities.SetInt(Props.SplatCount, m_GpuSortDistances.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes(k, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch(k, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            m_SorterArgs.inputKeys = m_GpuSortDistances;
            m_SorterArgs.inputValues = m_GpuSortKeys;
            m_SorterArgs.count = (uint)count;
            if (m_Sorter.Valid)
                m_SorterArgs.resources = GpuSorting.SupportResources.Load((uint)count);
        }

        bool resourcesAreSetUp => m_ShaderSplats != null && m_ShaderComposite != null && m_ShaderDebugPoints != null &&
                                  m_ShaderDebugBoxes != null && m_CSSplatUtilities != null && SystemInfo.supportsComputeShaders;

        public void EnsureMaterials()
        {
            if (m_MatSplats == null && resourcesAreSetUp)
            {
                m_MatSplats = new Material(m_ShaderSplats) {name = "GaussianSplats"};
                m_MatComposite = new Material(m_ShaderComposite) {name = "GaussianClearDstAlpha"};
                m_MatDebugPoints = new Material(m_ShaderDebugPoints) {name = "GaussianDebugPoints"};
                m_MatDebugBoxes = new Material(m_ShaderDebugBoxes) {name = "GaussianDebugBoxes"};
            }
        }

        public void EnsureSorterAndRegister()
        {
            if (m_Sorter == null && resourcesAreSetUp)
            {
                m_Sorter = new GpuSorting(m_CSSplatUtilities);
            }

            if (!m_Registered && resourcesAreSetUp)
            {
                GaussianSplatRenderSystem.instance.RegisterSplat(this);
                m_Registered = true;
            }
        }

        public void OnEnable()
        {

            m_FrameCounter = 0;
            if (!resourcesAreSetUp)
                return;

            EnsureMaterials();
            EnsureSorterAndRegister();

            CreateResourcesForAsset();
        }

        void SetAssetDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;

            //HJ
            int kernelIndex = K(kernel);
            //

            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSH, m_GpuSHData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatViewData, m_GpuView);
            
            //HJ
            //cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuSortKeys);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuOrderBuffer);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSortKeys, m_GpuSortKeys);

            cmb.SetComputeIntParam(cs, Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);

            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            //OOO
            cmb.SetComputeIntParam(cs, Props.UseFrameData, m_LastAppliedDeltaFrame >= 2 ? 1 : 0);
            //

            UpdateCutoutsBuffer();
            cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatCutouts, m_GpuEditCutouts);

            //HJ
            cmb.SetComputeBufferParam(cs, kernelIndex, "_GSBuffer", m_GpuGSBuffer);

            cmb.SetComputeBufferParam(cs, kernelIndex, "_FrameRot", m_GpuFrameRot);
            cmb.SetComputeBufferParam(cs, kernelIndex, "_FrameScale", m_GpuFrameScale);
            cmb.SetComputeBufferParam(cs, kernelIndex, "_FrameOpacity", m_GpuFrameOpacity);
            

        }

        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            mat.SetBuffer(Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            mat.SetBuffer(Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            mat.SetInt(Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);
            DisposeBuffer(ref m_GpuChunks);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuIndexBuffer);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);
            
            //HJ
            DisposeBuffer(ref m_GpuOrderBuffer);
            //

            DisposeBuffer(ref m_GpuEditSelectedMouseDown);
            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);
            DisposeBuffer(ref m_GpuEditSelected);
            DisposeBuffer(ref m_GpuEditDeleted);
            DisposeBuffer(ref m_GpuEditCountsBounds);
            DisposeBuffer(ref m_GpuEditCutouts);
            
            //HJ
            DisposeBuffer(ref m_GpuGSBuffer);
            DisposeBuffer(ref m_GpuDeltaIndices);
            DisposeBuffer(ref m_GpuDeltaXYZ);
            DisposeBuffer(ref m_GpuFrameRot);
            DisposeBuffer(ref m_GpuFrameScale);
            DisposeBuffer(ref m_GpuFrameOpacity);
            //

            m_SorterArgs.resources.Dispose();

            m_SplatCount = 0;
            m_GpuChunksValid = false;
            
            editSelectedSplats = 0;
            editDeletedSplats = 0;
            editCutSplats = 0;
            editModified = false;
            editSelectedBounds = default;

            //OOO
            m_GpuColorDataTex = null;
            m_TmpColorHalf = null;
            m_TmpPackedSH = null;
            m_TmpColorFloat = null;
            //
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);
            m_Registered = false;

            DestroyImmediate(m_MatSplats);
            DestroyImmediate(m_MatComposite);
            DestroyImmediate(m_MatDebugPoints);
            DestroyImmediate(m_MatDebugBoxes);
        }

        internal void CalcViewData(CommandBuffer cmb, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;
            int kernelView = K(KernelIndices.CalcViewData);

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.CalcViewData);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            //HJ
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            //
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);
            
            //SJ
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernelView, "_SplatSortKeys", m_GpuSortKeys);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kernelView, "_GSBuffer", m_GpuGSBuffer);
            //

            m_CSSplatUtilities.GetKernelThreadGroupSizes(kernelView, out uint gsX, out _, out _);

            cmb.DispatchCompute(m_CSSplatUtilities, kernelView, (m_GpuView.count + (int)gsX - 1)/(int)gsX, 1, 1);
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {


            if (cam.cameraType == CameraType.Preview)
                return;

            //HJ
            int kernelSet = K(KernelIndices.SetIndices);       // CSSetIndices
            int kernelDist = K(KernelIndices.CalcDistances);   // CSCalcDistances
            //

            Matrix4x4 worldToCamMatrix = cam.worldToCameraMatrix;
            worldToCamMatrix.m20 *= -1;
            worldToCamMatrix.m21 *= -1;
            worldToCamMatrix.m22 *= -1;

            cmd.BeginSample(s_ProfSort);

            //HJ //(int)KernelIndices.SetIndices = kernelSet
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelSet, Props.SplatSortKeys, m_GpuSortKeys);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelSet,Props.OrderBuffer, m_GpuOrderBuffer); 

            m_CSSplatUtilities.GetKernelThreadGroupSizes(kernelSet, out uint gsX0, out _, out _);
            cmd.DispatchCompute(m_CSSplatUtilities,kernelSet,(m_SplatCount + (int)gsX0 - 1) / (int)gsX0,1, 1);
            //

            //HJ //(int)KernelIndices.CalcDistances =  kernelDist
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, Props.SplatSortDistances, m_GpuSortDistances);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, Props.SplatSortKeys, m_GpuSortKeys);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, Props.OrderBuffer, m_GpuOrderBuffer); 

            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, Props.SplatChunks, m_GpuChunks);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, Props.SplatPos, m_GpuPosData);
            //

            //HJ
            //cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatFormat, (int)m_Asset.posFormat);
            uint format =
                (uint)m_Asset.posFormat |
                ((uint)m_Asset.scaleFormat << 8) |
                ((uint)m_Asset.shFormat << 16);

            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatFormat, (int)format);
            //

            cmd.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, worldToCamMatrix * matrix);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            
            //HJ //OOO
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.UseFrameData, m_LastAppliedDeltaFrame >= 2 ? 1 : 0);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, kernelDist, "_GSBuffer", m_GpuGSBuffer);
            //

            m_CSSplatUtilities.GetKernelThreadGroupSizes(kernelDist, out uint gsX1, out _, out _);
            cmd.DispatchCompute(m_CSSplatUtilities,kernelDist,(m_GpuSortDistances.count + (int)gsX1 - 1) / (int)gsX1,1, 1);

            EnsureSorterAndRegister();

            m_SorterArgs.inputKeys = m_GpuSortDistances; // distances are keys
            m_SorterArgs.inputValues = m_GpuSortKeys;    // indices are values
            m_SorterArgs.count = (uint)m_SplatCount;

            m_Sorter.Dispatch(cmd, m_SorterArgs);
            Graphics.CopyBuffer(m_GpuSortKeys, m_GpuOrderBuffer);

            cmd.EndSample(s_ProfSort);
        }

        public void Update()
        {

            //HJ //추가
            if (!Application.isPlaying)
                return;
            //

            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                if (resourcesAreSetUp)
                {
                    DisposeResourcesForAsset();
                    CreateResourcesForAsset();
                }
                else
                {
                    Debug.LogError($"{nameof(GaussianSplatRenderer)} component is not set up correctly (Resource references are missing), or platform does not support compute shaders");
                }
            }


            //HJ
            if (m_DeltaFrame <= 1)
            {
                m_DeltaFrame++;
                return;
            }

            UpdateDeltaFrame(m_DeltaFrame);
            ApplyDeltaSparseOnceOnGPU(m_DeltaFrame);
            ApplyDenseAttributesOnceOnGPU(m_DeltaFrame);
            
            m_LastAppliedDeltaFrame = m_DeltaFrame;


            if (m_DeltaFrame < 300)
                m_DeltaFrame++;
        
        }
        

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }

        void ClearGraphicsBuffer(GraphicsBuffer buf)
        {
            if (buf == null || m_CSSplatUtilities == null) return;

            int k = K(KernelIndices.ClearBuffer);
            m_CSSplatUtilities.SetBuffer(k, Props.DstBuffer, buf);
            m_CSSplatUtilities.SetInt(Props.BufferSize, buf.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes(k, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch(k, (int)((buf.count + gsX - 1)/gsX), 1, 1);
        }

        void UnionGraphicsBuffers(GraphicsBuffer dst, GraphicsBuffer src)
        {
            if (dst == null || src == null || m_CSSplatUtilities == null) return;

            int k = K(KernelIndices.OrBuffers);
            if (k < 0) return;

            m_CSSplatUtilities.SetBuffer(k, Props.SrcBuffer, src);
            m_CSSplatUtilities.SetBuffer(k, Props.DstBuffer, dst);
            m_CSSplatUtilities.SetInt(Props.BufferSize, dst.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes(k, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch(k, (int)((dst.count + gsX - 1)/gsX), 1, 1);
        }

        static float SortableUintToFloat(uint v)
        {
            uint mask = ((v >> 31) - 1) | 0x80000000u;
            return math.asfloat(v ^ mask);
        }

        public void UpdateEditCountsAndBounds()
        {
            if (m_GpuEditSelected == null)
            {
                editSelectedSplats = 0;
                editDeletedSplats = 0;
                editCutSplats = 0;
                editModified = false;
                editSelectedBounds = default;
                return;
            }

            int kInit = K(KernelIndices.InitEditData);
            m_CSSplatUtilities.SetBuffer(kInit, Props.DstBuffer, m_GpuEditCountsBounds);
            m_CSSplatUtilities.Dispatch(kInit, 1, 1, 1);

            using CommandBuffer cmb = new CommandBuffer();
            int kUpdate = K(KernelIndices.UpdateEditData);
            SetAssetDataOnCS(cmb, KernelIndices.UpdateEditData);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kUpdate, Props.DstBuffer, m_GpuEditCountsBounds);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes(kUpdate, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, kUpdate, (int)((m_GpuEditSelected.count + gsX - 1) / gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

            uint[] res = new uint[m_GpuEditCountsBounds.count];
            m_GpuEditCountsBounds.GetData(res);
            editSelectedSplats = res[0];
            editDeletedSplats = res[1];
            editCutSplats = res[2];
            Vector3 min = new Vector3(SortableUintToFloat(res[3]), SortableUintToFloat(res[4]), SortableUintToFloat(res[5]));
            Vector3 max = new Vector3(SortableUintToFloat(res[6]), SortableUintToFloat(res[7]), SortableUintToFloat(res[8]));
            Bounds bounds = default;
            bounds.SetMinMax(min, max);
            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f, 0.1f, 0.1f);
            editSelectedBounds = bounds;
        }

        void UpdateCutoutsBuffer()
        {
            int bufferSize = m_Cutouts?.Length ?? 0;
            if (bufferSize == 0)
                bufferSize = 1;
            if (m_GpuEditCutouts == null || m_GpuEditCutouts.count != bufferSize)
            {
                m_GpuEditCutouts?.Dispose();
                m_GpuEditCutouts = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, UnsafeUtility.SizeOf<GaussianCutout.ShaderData>()) { name = "GaussianCutouts" };
            }

            NativeArray<GaussianCutout.ShaderData> data = new(bufferSize, Allocator.Temp);
            if (m_Cutouts != null)
            {
                var matrix = transform.localToWorldMatrix;
                for (var i = 0; i < m_Cutouts.Length; ++i)
                {
                    data[i] = GaussianCutout.GetShaderData(m_Cutouts[i], matrix);
                }
            }

            m_GpuEditCutouts.SetData(data);
            data.Dispose();
        }

        bool EnsureEditingBuffers()
        {
            if (!HasValidAsset || !HasValidRenderSetup)
                return false;

            if (m_GpuEditSelected == null)
            {
                var target = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource |
                             GraphicsBuffer.Target.CopyDestination;
                var size = (m_SplatCount + 31) / 32;
                m_GpuEditSelected = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelected"};
                m_GpuEditSelectedMouseDown = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatSelectedInit"};
                m_GpuEditDeleted = new GraphicsBuffer(target, size, 4) {name = "GaussianSplatDeleted"};
                m_GpuEditCountsBounds = new GraphicsBuffer(target, 3 + 6, 4) {name = "GaussianSplatEditData"}; // selected count, deleted bound, cut count, float3 min, float3 max
                ClearGraphicsBuffer(m_GpuEditSelected);
                ClearGraphicsBuffer(m_GpuEditSelectedMouseDown);
                ClearGraphicsBuffer(m_GpuEditDeleted);
            }
            return m_GpuEditSelected != null;
        }

        public void EditStoreSelectionMouseDown()
        {
            if (!EnsureEditingBuffers()) return;
            Graphics.CopyBuffer(m_GpuEditSelected, m_GpuEditSelectedMouseDown);
        }

        public void EditStorePosMouseDown()
        {
            if (m_GpuEditPosMouseDown == null)
            {
                m_GpuEditPosMouseDown = new GraphicsBuffer(m_GpuPosData.target | GraphicsBuffer.Target.CopyDestination, m_GpuPosData.count, m_GpuPosData.stride) {name = "GaussianSplatEditPosMouseDown"};
            }
            Graphics.CopyBuffer(m_GpuPosData, m_GpuEditPosMouseDown);
        }
        public void EditStoreOtherMouseDown()
        {
            if (m_GpuEditOtherMouseDown == null)
            {
                m_GpuEditOtherMouseDown = new GraphicsBuffer(m_GpuOtherData.target | GraphicsBuffer.Target.CopyDestination, m_GpuOtherData.count, m_GpuOtherData.stride) {name = "GaussianSplatEditOtherMouseDown"};
            }
            Graphics.CopyBuffer(m_GpuOtherData, m_GpuEditOtherMouseDown);
        }

        public void EditUpdateSelection(Vector2 rectMin, Vector2 rectMax, Camera cam, bool subtract)
        {
            if (!EnsureEditingBuffers()) return;

            Graphics.CopyBuffer(m_GpuEditSelectedMouseDown, m_GpuEditSelected);

            var tr = transform;
            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            using var cmb = new CommandBuffer { name = "SplatSelectionUpdate" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectionUpdate);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_SelectionRect", new Vector4(rectMin.x, rectMax.y, rectMax.x, rectMin.y));
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SelectionMode, subtract ? 0 : 1);

            DispatchUtilsAndExecute(cmb, KernelIndices.SelectionUpdate, m_SplatCount);
            UpdateEditCountsAndBounds();
        }

        public void EditTranslateSelection(Vector3 localSpacePosDelta)
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatTranslateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.TranslateSelection);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, localSpacePosDelta);

            DispatchUtilsAndExecute(cmb, KernelIndices.TranslateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditRotateSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Quaternion rotation)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null || m_GpuEditOtherMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatRotateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.RotateSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, K(KernelIndices.RotateSelection), Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, K(KernelIndices.RotateSelection), Props.SplatOtherMouseDown, m_GpuEditOtherMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDeltaRot, new Vector4(rotation.x, rotation.y, rotation.z, rotation.w));

            DispatchUtilsAndExecute(cmb, KernelIndices.RotateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }


        public void EditScaleSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Vector3 scale)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatScaleSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.ScaleSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, K(KernelIndices.ScaleSelection), Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, scale);

            DispatchUtilsAndExecute(cmb, KernelIndices.ScaleSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditDeleteSelected()
        {
            if (!EnsureEditingBuffers()) return;
            UnionGraphicsBuffers(m_GpuEditDeleted, m_GpuEditSelected);
            EditDeselectAll();
            UpdateEditCountsAndBounds();
            if (editDeletedSplats != 0)
                editModified = true;
        }

        public void EditSelectAll()
        {
            if (!EnsureEditingBuffers()) return;
            using var cmb = new CommandBuffer { name = "SplatSelectAll" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectAll);

            int kSelectAll = K(KernelIndices.SelectAll);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kSelectAll, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);

            DispatchUtilsAndExecute(cmb, KernelIndices.SelectAll, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public void EditDeselectAll()
        {
            if (!EnsureEditingBuffers()) return;
            ClearGraphicsBuffer(m_GpuEditSelected);
            UpdateEditCountsAndBounds();
        }

        public void EditInvertSelection()
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatInvertSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.InvertSelection);

            int kInvert = K(KernelIndices.InvertSelection);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kInvert, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);

            DispatchUtilsAndExecute(cmb, KernelIndices.InvertSelection, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public bool EditExportData(GraphicsBuffer dstData, bool bakeTransform)
        {
            if (!EnsureEditingBuffers()) return false;

            int flags = 0;
            var tr = transform;
            Quaternion bakeRot = tr.localRotation;
            Vector3 bakeScale = tr.localScale;

            if (bakeTransform)
                flags = 1;

            using var cmb = new CommandBuffer { name = "SplatExportData" };
            SetAssetDataOnCS(cmb, KernelIndices.ExportData);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_ExportTransformFlags", flags);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformRotation", new Vector4(bakeRot.x, bakeRot.y, bakeRot.z, bakeRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformScale", bakeScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, tr.localToWorldMatrix);

            int kExport = K(KernelIndices.ExportData);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kExport, "_ExportBuffer", dstData);

            DispatchUtilsAndExecute(cmb, KernelIndices.ExportData, m_SplatCount);
            return true;
        }

        public void EditSetSplatCount(int newSplatCount)
        {
            if (newSplatCount <= 0 || newSplatCount > GaussianSplatAsset.kMaxSplats)
            {
                Debug.LogError($"Invalid new splat count: {newSplatCount}");
                return;
            }
            if (asset.chunkData != null)
            {
                Debug.LogError("Only splats with VeryHigh quality can be resized");
                return;
            }
            if (newSplatCount == splatCount)
                return;

            int posStride = (int)(asset.posData.dataSize / asset.splatCount);
            int otherStride = (int)(asset.otherData.dataSize / asset.splatCount);
            int shStride = (int) (asset.shData.dataSize / asset.splatCount);

            // create new GPU buffers
            var newPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * posStride / 4, 4) { name = "GaussianPosData" };
            var newOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * otherStride / 4, 4) { name = "GaussianOtherData" };
            var newSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newSplatCount * shStride / 4, 4) { name = "GaussianSHData" };

            // new texture is a RenderTexture so we can write to it from a compute shader
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(newSplatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var newColorData = new RenderTexture(texWidth, texHeight, texFormat, GraphicsFormat.None) { name = "GaussianColorData", enableRandomWrite = true };
            newColorData.Create();

            // selected/deleted buffers
            var selTarget = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource | GraphicsBuffer.Target.CopyDestination;
            var selSize = (newSplatCount + 31) / 32;
            var newEditSelected = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatSelected"};
            var newEditSelectedMouseDown = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatSelectedInit"};
            var newEditDeleted = new GraphicsBuffer(selTarget, selSize, 4) {name = "GaussianSplatDeleted"};
            ClearGraphicsBuffer(newEditSelected);
            ClearGraphicsBuffer(newEditSelectedMouseDown);
            ClearGraphicsBuffer(newEditDeleted);

            var newGpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, newSplatCount, kGpuViewDataSize);
            InitSortBuffers(newSplatCount);

            // copy existing data over into new buffers
            EditCopySplats(transform, newPosData, newOtherData, newSHData, newColorData, newEditDeleted, newSplatCount, 0, 0, m_SplatCount);

            // use the new buffers and the new splat count
            m_GpuPosData.Dispose();
            m_GpuOtherData.Dispose();
            m_GpuSHData.Dispose();
            DestroyImmediate(m_GpuColorData);
            m_GpuView.Dispose();

            m_GpuEditSelected?.Dispose();
            m_GpuEditSelectedMouseDown?.Dispose();
            m_GpuEditDeleted?.Dispose();

            m_GpuPosData = newPosData;
            m_GpuOtherData = newOtherData;
            m_GpuSHData = newSHData;
            m_GpuColorData = newColorData;
            m_GpuView = newGpuView;
            m_GpuEditSelected = newEditSelected;
            m_GpuEditSelectedMouseDown = newEditSelectedMouseDown;
            m_GpuEditDeleted = newEditDeleted;

            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);

            m_SplatCount = newSplatCount;
            editModified = true;
        }

        public void EditCopySplatsInto(GaussianSplatRenderer dst, int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            EditCopySplats(
                dst.transform,
                dst.m_GpuPosData, dst.m_GpuOtherData, dst.m_GpuSHData, dst.m_GpuColorData, dst.m_GpuEditDeleted,
                dst.splatCount,
                copySrcStartIndex, copyDstStartIndex, copyCount);
            dst.editModified = true;
        }

        public void EditCopySplats(
            Transform dstTransform,
            GraphicsBuffer dstPos, GraphicsBuffer dstOther, GraphicsBuffer dstSH, Texture dstColor,
            GraphicsBuffer dstEditDeleted,
            int dstSize,
            int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            if (!EnsureEditingBuffers()) return;

            Matrix4x4 copyMatrix = dstTransform.worldToLocalMatrix * transform.localToWorldMatrix;
            Quaternion copyRot = copyMatrix.rotation;
            Vector3 copyScale = copyMatrix.lossyScale;

            using var cmb = new CommandBuffer { name = "SplatCopy" };
            SetAssetDataOnCS(cmb, KernelIndices.CopySplats);

            int kCopy = K(KernelIndices.CopySplats);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, kCopy, "_CopyDstPos", dstPos);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kCopy, "_CopyDstOther", dstOther);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kCopy, "_CopyDstSH", dstSH);
            cmb.SetComputeTextureParam(m_CSSplatUtilities, kCopy, "_CopyDstColor", dstColor);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, kCopy, "_CopyDstEditDeleted", dstEditDeleted);

            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstSize", dstSize);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopySrcStartIndex", copySrcStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstStartIndex", copyDstStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyCount", copyCount);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformRotation", new Vector4(copyRot.x, copyRot.y, copyRot.z, copyRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformScale", copyScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, "_CopyTransformMatrix", copyMatrix);

            DispatchUtilsAndExecute(cmb, KernelIndices.CopySplats, copyCount);
        }

        void DispatchUtilsAndExecute(CommandBuffer cmb, KernelIndices kernel, int count)
        {
            int k = K(kernel);
            m_CSSplatUtilities.GetKernelThreadGroupSizes(k, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, k, (int)((count + gsX - 1) / gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);
        }

        //HJ

        public unsafe void UpdateDeltaFrame(int frameNumber)
        {   
            if (frameNumber <= 1)
            {
                return;
            }
            string binPath = Path.Combine(deltaRoot, $"{frameNumber:D4}.bin");

            //OOO
            var frame = hj_read_bin_float32.Read(binPath);
            if (frame == null)
                throw new InvalidOperationException($"Failed to read frame bin: {binPath}");

            if (frame.header.N < m_SplatCount)
                throw new InvalidDataException($"Frame N too small. bin={frame.header.N}, asset={m_SplatCount}");

            if (frame.header.N != m_SplatCount)
                Debug.LogWarning($"[BIN] N mismatch. bin={frame.header.N}, asset={m_SplatCount}. Extra tail will be ignored.");
                
            if (frame.header.fdcDim != kFrameFdcDim)
                throw new InvalidDataException($"f_dc dim must be {kFrameFdcDim}, got {frame.header.fdcDim}");

            if (frame.header.frestDim != kFrameFRestDim)
                throw new InvalidDataException($"f_rest dim must be {kFrameFRestDim}, got {frame.header.frestDim}");
            
            m_TmpIndices = frame.indices;
            m_TmpDeltaXYZ = frame.xyzDelta;
            m_TmpRot = frame.rot;
            m_TmpScale = frame.scale;
            m_TmpOpacity = frame.opacity;
            //

            int K = (m_TmpIndices != null) ? m_TmpIndices.Length : 0;

            int validK  = 0; //VALID 한 값들만
            if (K > 0 && m_TmpDeltaXYZ != null && m_TmpDeltaXYZ.Length == K)
            {
                var validI = new List<int>(K);
                var validD = new List<Vector3>(K);

                for (int i = 0; i < K; i++)
                {
                    int idx = m_TmpIndices[i];
                    if (idx < 0 || idx >= m_SplatCount) continue;

                    validI.Add(idx);
                    validD.Add(m_TmpDeltaXYZ[i]);
                }

                validK  = validI.Count;
    
                if (validK > 0)
                {
                    m_GpuDeltaIndices.SetData(validI, 0, 0, validK);
                    m_GpuDeltaXYZ.SetData(validD, 0, 0, validK);
                }
            }

            m_LastDeltaK = validK;
            
            //OOO
            UploadFrameColor(frame.fdc, frame.opacity);
            UploadFrameSH(frame.frest);
            

            var rotVec = new Vector4[m_SplatCount];
            for (int i = 0; i < m_SplatCount; i++)
            {
                Quaternion q = m_TmpRot[i];
                rotVec[i] = new Vector4(q.x, q.y, q.z, q.w); //이 순서 맞아.
            }
            m_GpuFrameRot.SetData(rotVec, 0, 0, m_SplatCount);

            m_GpuFrameScale.SetData(frame.scale, 0, 0, m_SplatCount);
            m_GpuFrameOpacity.SetData(frame.opacity, 0, 0, m_SplatCount);

            m_LastLoadedDeltaFrame = frameNumber;
            //
        }

            //OOO
            void UploadFrameColor(Vector3[] fdc, float[] opacity)
            {
                if (m_GpuColorDataTex == null)
                    throw new InvalidOperationException("m_GpuColorDataTex is null.");

                if (fdc == null || fdc.Length < m_SplatCount)
                    throw new InvalidDataException($"f_dc length mismatch: {fdc?.Length ?? 0} < {m_SplatCount}");

                if (opacity == null || opacity.Length < m_SplatCount)
                    throw new InvalidDataException($"opacity length mismatch: {opacity?.Length ?? 0} < {m_SplatCount}");

                int texWidth = m_GpuColorDataTex.width;
                int texHeight = m_GpuColorDataTex.height;
                int valueCount = texWidth * texHeight * 4;

                // BIN의 f_dc는 SH0 coefficient 이므로 최종 color로 역변환
                const float kSH0 = 0.2820948f;

                switch (m_GpuColorDataTex.graphicsFormat)
                {
                    case GraphicsFormat.R16G16B16A16_SFloat:
                    {
                        if (m_TmpColorHalf == null || m_TmpColorHalf.Length != valueCount)
                            m_TmpColorHalf = new ushort[valueCount];

                        Array.Clear(m_TmpColorHalf, 0, m_TmpColorHalf.Length);

                        for (int i = 0; i < m_SplatCount; i++)
                        {
                            SplatIndexToPixelCoord(i, texWidth, out int x, out int y);
                            int p = (y * texWidth + x) * 4;

                            Vector3 dc = fdc[i];
                            Vector3 col = dc * kSH0 + new Vector3(0.5f, 0.5f, 0.5f);

                            // color texture alpha는 "최종 opacity" 쪽이 의미상 맞음
                            float a = 1.0f / (1.0f + Mathf.Exp(-opacity[i]));
                            a = Mathf.Clamp01(a);

                            m_TmpColorHalf[p + 0] = Mathf.FloatToHalf(col.x);
                            m_TmpColorHalf[p + 1] = Mathf.FloatToHalf(col.y);
                            m_TmpColorHalf[p + 2] = Mathf.FloatToHalf(col.z);
                            m_TmpColorHalf[p + 3] = Mathf.FloatToHalf(a);
                        }

                        m_GpuColorDataTex.SetPixelData(m_TmpColorHalf, 0);
                        break;
                    }

                    case GraphicsFormat.R32G32B32A32_SFloat:
                    {
                        if (m_TmpColorFloat == null || m_TmpColorFloat.Length != valueCount)
                            m_TmpColorFloat = new float[valueCount];

                        Array.Clear(m_TmpColorFloat, 0, m_TmpColorFloat.Length);

                        for (int i = 0; i < m_SplatCount; i++)
                        {
                            SplatIndexToPixelCoord(i, texWidth, out int x, out int y);
                            int p = (y * texWidth + x) * 4;

                            Vector3 dc = fdc[i];
                            Vector3 col = dc * kSH0 + new Vector3(0.5f, 0.5f, 0.5f);

                            float a = 1.0f / (1.0f + Mathf.Exp(-opacity[i]));
                            a = Mathf.Clamp01(a);

                            m_TmpColorFloat[p + 0] = col.x;
                            m_TmpColorFloat[p + 1] = col.y;
                            m_TmpColorFloat[p + 2] = col.z;
                            m_TmpColorFloat[p + 3] = a;
                        }

                        m_GpuColorDataTex.SetPixelData(m_TmpColorFloat, 0);
                        break;
                    }

                    case GraphicsFormat.R8G8B8A8_UNorm:
                    case GraphicsFormat.RGBA_BC7_UNorm:
                        throw new NotSupportedException(
                            $"UploadFrameColor does not support runtime upload to {m_GpuColorDataTex.graphicsFormat}. " +
                            $"Use a runtime Float16x4 or Float32x4 color texture for frame updates.");

                    default:
                        throw new NotSupportedException(
                            $"Unsupported color texture format: {m_GpuColorDataTex.graphicsFormat}");
                }

                m_GpuColorDataTex.Apply(false, false);
            }

            void UploadFrameSH(float[] frest)
        {
            if (frest == null || frest.Length < m_SplatCount * kFrameFRestDim)
                throw new InvalidDataException(
                    $"f_rest length mismatch: {frest?.Length ?? 0} < {m_SplatCount * kFrameFRestDim}");

            if (asset.shFormat != GaussianSplatAsset.SHFormat.Float32)
                throw new NotSupportedException(
                    $"Current UploadFrameSH only supports asset.shFormat == Float32, but got {asset.shFormat}");

            const int dstStrideU32 = 48; // Float32 SHTableItem: 15 * float3 + padding
            int requiredU32 = m_SplatCount * dstStrideU32;

            if (m_GpuSHData.count != requiredU32)
                throw new InvalidDataException(
                    $"_SplatSH raw size mismatch: gpu={m_GpuSHData.count}, expected={requiredU32}, shFormat={asset.shFormat}");

            if (m_TmpPackedSH == null || m_TmpPackedSH.Length != requiredU32)
                m_TmpPackedSH = new uint[requiredU32];

            Array.Clear(m_TmpPackedSH, 0, m_TmpPackedSH.Length);

            // BIN frest(24) = [sh1.r, sh1.g, sh1.b, sh2.r, sh2.g, sh2.b, ... , sh8.r, sh8.g, sh8.b]
            for (int i = 0; i < m_SplatCount; i++)
            {
                int src = i * kFrameFRestDim;
                int dst = i * dstStrideU32;

                for (int c = 0; c < 8; c++)
                {
                    int s = src + c * 3;

                    float r = frest[s + 0];
                    float g = frest[s + 1];
                    float b = frest[s + 2];

                    int o = dst + c * 3;
                    m_TmpPackedSH[o + 0] = math.asuint(r);
                    m_TmpPackedSH[o + 1] = math.asuint(g);
                    m_TmpPackedSH[o + 2] = math.asuint(b);
                }

                // sh9 ~ sh15 + padding = 0 유지
            }

            m_GpuSHData.SetData(m_TmpPackedSH, 0, 0, m_TmpPackedSH.Length);
        }
        //

        static void SplatIndexToPixelCoord(int splatIndex, int texWidth, out int x, out int y)
        {
            uint morton = (uint)(splatIndex & 255);
            int localX = (int)Compact1By1(morton);
            int localY = (int)Compact1By1(morton >> 1);

            int tileIndex = splatIndex >> 8;
            int tilesPerRow = texWidth / 16;

            x = (tileIndex % tilesPerRow) * 16 + localX;
            y = (tileIndex / tilesPerRow) * 16 + localY;
        }

        static uint Compact1By1(uint x)
        {
            x &= 0x55555555u;
            x = (x ^ (x >> 1)) & 0x33333333u;
            x = (x ^ (x >> 2)) & 0x0f0f0f0fu;
            x = (x ^ (x >> 4)) & 0x00ff00ffu;
            x = (x ^ (x >> 8)) & 0x0000ffffu;
            return x;
        }
    }
}
