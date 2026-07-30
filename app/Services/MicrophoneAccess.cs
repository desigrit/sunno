using System;
using System.Threading.Tasks;
using Windows.Security.Authorization.AppCapabilityAccess;

namespace Sunno.Services;

/// <summary>
/// Microphone consent, as Windows actually models it.
///
/// The distinction that matters: <c>CheckAccess()</c> only reports the stored decision and is
/// documented to never prompt, while <c>RequestAccessAsync()</c> asks the system for a real
/// access check and is what raises the consent dialog. Conflating the two — and treating
/// "never asked" (<c>UserPromptRequired</c>) as a refusal — is why this app used to send
/// people to Settings for a permission they had never been offered.
///
/// None of this exists without package identity, which is the concrete payoff of shipping as
/// MSIX: an unpackaged app has no per-app microphone toggle and no way to raise this dialog.
/// </summary>
internal static class MicrophoneAccess
{
    private static AppCapability? _capability;
    private static bool _resolved;
    private static Action? _changed;
    private static bool _hooked;

    /// <summary>
    /// The capability object, or null when it cannot exist. Cached because an AccessChanged
    /// subscription lives on the instance, so a fresh object per call would drop the hook.
    /// </summary>
    private static AppCapability? Capability
    {
        get
        {
            if (_resolved) return _capability;
            _resolved = true;

            // Checked rather than caught: AppCapability requires package identity, and an
            // unpackaged dev build hitting this is expected, not exceptional.
            if (!BackendHost.IsPackaged()) return null;

            try
            {
                _capability = AppCapability.Create("microphone");
            }
            catch (Exception ex)
            {
                // Identity present but the API absent — an OS older than the manifest floor.
                System.Diagnostics.Debug.WriteLine($"AppCapability unavailable: {ex.Message}");
            }
            return _capability;
        }
    }

    /// <summary>True when consent is a concept here at all, i.e. we are running packaged.</summary>
    public static bool IsSupported => Capability is not null;

    /// <summary>
    /// Read the stored decision. Guaranteed not to prompt, so this is safe to call during
    /// startup to decide whether the microphone can be opened yet. Null means unsupported.
    /// </summary>
    public static AppCapabilityAccessStatus? Check()
    {
        try
        {
            return Capability?.CheckAccess();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CheckAccess failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Ask Windows for access, raising the consent dialog when the user has not been asked
    /// before. Documented to never return <c>UserPromptRequired</c>.
    ///
    /// The API reference states this <b>must be called from the UI thread</b>, so callers must
    /// not hop off it first — no <c>Task.Run</c>, and no <c>ConfigureAwait(false)</c> on any
    /// await preceding it, or the dialog silently fails to appear.
    /// </summary>
    public static async Task<AppCapabilityAccessStatus?> RequestAsync()
    {
        var capability = Capability;
        if (capability is null) return null;

        try
        {
            return await capability.RequestAccessAsync();
        }
        catch (Exception ex)
        {
            // Fall back to the stored decision so the caller still renders something truthful
            // rather than treating a failed request as a denial.
            System.Diagnostics.Debug.WriteLine($"RequestAccessAsync failed: {ex.Message}");
            return Check();
        }
    }

    /// <summary>
    /// Raised when consent changes underneath the app — typically the user flipping the toggle
    /// in Settings while the window is open. May arrive on a background thread, so handlers
    /// must marshal to the UI thread themselves.
    /// </summary>
    public static event Action Changed
    {
        add
        {
            var capability = Capability;
            if (capability is null) return;

            _changed += value;
            if (_hooked) return;
            _hooked = true;
            capability.AccessChanged += OnAccessChanged;
        }
        remove
        {
            _changed -= value;
            if (_changed is not null || !_hooked) return;

            // Drop the WinRT hook once nothing is listening, so a closed window cannot be
            // resurrected by a late callback.
            _hooked = false;
            if (Capability is { } capability) capability.AccessChanged -= OnAccessChanged;
        }
    }

    private static void OnAccessChanged(AppCapability sender, AppCapabilityAccessChangedEventArgs args)
        => _changed?.Invoke();
}
