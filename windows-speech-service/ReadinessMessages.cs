internal static class ReadinessMessages
{
    public static string ForState(string state) => state switch
    {
        "Ready" => "",
        "NotReady" => "Windows must install its speech model. Choose this model to set it up.",
        "DisabledByUser" => "Windows AI models are disabled. Enable them in Windows Settings or ask your administrator.",
        "CapabilityMissing" => "Windows denied access to AI models. Install the latest Sunno package and restart the app.",
        "OSUpdateNeeded" => "Windows AI speech needs a Windows update. Install pending updates in Settings > Windows Update, then restart Sunno.",
        "NotCompatibleWithSystemHardware" => "The Windows speech model is not compatible with this PC. Choose Moonshine or Whisper instead.",
        "NotSupportedOnCurrentSystem" => "Windows AI speech is unavailable on this version of Windows. Update Windows or choose Moonshine or Whisper.",
        _ => $"Windows speech model status: {state}. Check Settings > Windows Update or choose another model.",
    };

    public static string ForSetup(string state, string status, string displayText,
                                  string error, string extendedError, bool packageFailed)
    {
        if (state != "NotReady") return ForState(state);
        if (status != "Failure")
            return "Windows is still installing its speech model. Check Settings > Windows Update, then choose this model again when installation finishes.";
        var details = new[] { displayText, error, extendedError }
            .Where(value => !string.IsNullOrWhiteSpace(value)).Distinct();
        var detail = string.Join("; ", details);
        var problem = packageFailed ? "Windows could not install its speech model" : "Windows could not set up its speech model";
        return $"{problem}{(detail.Length > 0 ? $": {detail}" : "")}. Check Settings > Windows Update for pending or failed downloads, then retry. CPU-only Intel PCs are supported.";
    }
}
