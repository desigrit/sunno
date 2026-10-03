using System.Runtime.InteropServices;

namespace Sunno.Services;

/// <summary>Core Audio callbacks only enqueue work; no enumeration runs inside a callback.</summary>
public sealed class AudioEndpointWatcher : IDisposable
{
    private IDeviceEnumerator? _enumerator;
    private readonly Notifications _notifications;
    public event Action? Changed;

    public AudioEndpointWatcher()
    {
        _notifications = new Notifications(() => Changed?.Invoke());
        _enumerator = (IDeviceEnumerator)new DeviceEnumerator();
        try { Marshal.ThrowExceptionForHR(_enumerator.RegisterEndpointNotificationCallback(_notifications)); }
        catch { Marshal.FinalReleaseComObject(_enumerator); _enumerator = null; throw; }
    }

    public void Dispose()
    {
        var enumerator = Interlocked.Exchange(ref _enumerator, null);
        if (enumerator is null) return;
        Changed = null;
        try { enumerator.UnregisterEndpointNotificationCallback(_notifications); }
        finally { Marshal.FinalReleaseComObject(enumerator); }
    }

    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    private class DeviceEnumerator { }

    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDeviceEnumerator
    {
        [PreserveSig] int EnumAudioEndpoints(int flow, uint states, out IntPtr devices);
        [PreserveSig] int GetDefaultAudioEndpoint(int flow, int role, out IntPtr device);
        [PreserveSig] int GetDevice([MarshalAs(UnmanagedType.LPWStr)] string id, out IntPtr device);
        [PreserveSig] int RegisterEndpointNotificationCallback(INotifications callback);
        [PreserveSig] int UnregisterEndpointNotificationCallback(INotifications callback);
    }

    [ComVisible(true), Guid("7991EEC9-7E89-4D85-8390-6C703CEC60C0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INotifications
    {
        [PreserveSig] int OnDeviceStateChanged([MarshalAs(UnmanagedType.LPWStr)] string id, uint state);
        [PreserveSig] int OnDeviceAdded([MarshalAs(UnmanagedType.LPWStr)] string id);
        [PreserveSig] int OnDeviceRemoved([MarshalAs(UnmanagedType.LPWStr)] string id);
        [PreserveSig] int OnDefaultDeviceChanged(int flow, int role, [MarshalAs(UnmanagedType.LPWStr)] string? id);
        [PreserveSig] int OnPropertyValueChanged([MarshalAs(UnmanagedType.LPWStr)] string id, PropertyKey key);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PropertyKey { public Guid FormatId; public uint PropertyId; }

    [ComVisible(true), ClassInterface(ClassInterfaceType.None)]
    private sealed class Notifications(Action changed) : INotifications
    {
        private int Signal() { try { changed(); } catch { } return 0; }
        public int OnDeviceStateChanged(string id, uint state) => Signal();
        public int OnDeviceAdded(string id) => Signal();
        public int OnDeviceRemoved(string id) => Signal();
        public int OnDefaultDeviceChanged(int flow, int role, string? id) => Signal();
        public int OnPropertyValueChanged(string id, PropertyKey key) => Signal();
    }
}
