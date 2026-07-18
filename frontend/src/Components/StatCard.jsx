function StatCard({
  icon,
  iconSize,
  iconColor,
  borderColor,
  label,
  value,
  valueTag: ValueTag = "h4",
  valueColor,
  percent,
}) {
  return (
    <div className="card h-100" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header p-3 pt-2">
        <div className="d-flex align-items-center">
          <i className={icon} style={{ fontSize: iconSize, color: iconColor }} aria-hidden="true"></i>
          <p className="text-sm mb-0 text-capitalize">{label}</p>
        </div>
        <ValueTag className="mb-0 mt-2" style={valueColor ? { color: valueColor } : undefined}>
          {value}
        </ValueTag>
        {percent && <p className="text-xs text-muted mb-0">{percent}</p>}
      </div>
      <hr className="dark horizontal my-0" />
    </div>
  );
}

export default StatCard;
