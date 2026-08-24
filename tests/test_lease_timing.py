from mc_ai_bt.lease_timing import lease_renew_interval


def test_lease_renew_interval_has_margin_before_ttl_expiry():
    assert lease_renew_interval(300.0) == 30.0
    assert lease_renew_interval(10.0) == 5.0
    assert lease_renew_interval(0.1) == 1.0
