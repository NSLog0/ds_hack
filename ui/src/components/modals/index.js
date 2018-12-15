import React from 'react';
import classNames from 'classnames';

class Modal extends React.Component {
  render() {
    return (
      <div
        className={classNames('modal', { 'modal--active': this.props.isShow })}
      >
        <div className="modal__dialog">
          <div className="modal__header">
            <h2>{this.props.title}</h2>
            <span onClick={this.props.close} className="btn--close">
              ×
            </span>
          </div>
          <div className="modal__body">{this.props.children}</div>
          <div className="modal__footer">
            <button className={'btn btn--login'} onClick={this.props.action}>
              {this.props.text}
            </button>
          </div>
        </div>
      </div>
    );
  }
}
export default Modal;
