using System.Runtime.InteropServices;

namespace Sunno.Services;

/// <summary>
/// Ties child processes to the lifetime of this process using a Windows Job Object.
///
/// Without this, killing or crashing the UI leaves the Python backend running — still holding
/// the microphone open. For an app that records other people, a capture process that outlives
/// its visible window is a privacy defect, not just untidy bookkeeping. The kernel enforces
/// this teardown, so it survives paths that never run managed cleanup.
/// </summary>
public sealed class ChildProcessJob : IDisposable
{
    private const int JobObjectExtendedLimitInfoClass = 9;
    private const uint JobObjectLimitKillOnJobClose = 0x2000;

    private IntPtr _handle;

    public ChildProcessJob()
    {
        _handle = CreateJobObject(IntPtr.Zero, null);
        if (_handle == IntPtr.Zero) return;

        var info = new JobObjectBasicLimitInformation { LimitFlags = JobObjectLimitKillOnJobClose };
        var extended = new JobObjectExtendedLimitInformation { BasicLimitInformation = info };

        var length = Marshal.SizeOf<JobObjectExtendedLimitInformation>();
        var ptr = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(extended, ptr, false);
            SetInformationJobObject(_handle, JobObjectExtendedLimitInfoClass, ptr, (uint)length);
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }

    public bool Assign(System.Diagnostics.Process process)
    {
        if (_handle == IntPtr.Zero) return false;
        try { return AssignProcessToJobObject(_handle, process.Handle); }
        catch { return false; }
    }

    public void Dispose()
    {
        if (_handle == IntPtr.Zero) return;
        CloseHandle(_handle);   // closing the last handle kills everything in the job
        _handle = IntPtr.Zero;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformation
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string? name);

    [DllImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);

    [DllImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);
}
