try:
    import metalcompute as mc
    device = mc.Device()
    print(f"Metal Device: {device.name}")
except Exception as e:
    print(f"Metal check failed: {str(e)}")
